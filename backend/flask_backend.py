from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import traceback

from virtual_color_mixing_lab import VirtualColorMixingLab

# Optional colour-science dependency
import numpy as np
import colour

app = Flask(__name__)
CORS(app)

sessions = {}

CHANNEL_WAVELENGTHS_NM = {
    'ch410': 410,
    'ch440': 440,
    'ch470': 470,
    'ch510': 510,
    'ch550': 550,
    'ch583': 583,
    'ch620': 620,
    'ch670': 670,
}

def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def rgb01_to_hex(rgb01):
    r, g, b = [int(round(_clamp01(v) * 255)) for v in rgb01]
    return f"#{r:02X}{g:02X}{b:02X}"

def sensor_channels_to_srgb_hex(ch: dict, normalise: str = 'sum'):
    """Convert 8-channel sensor counts to an sRGB hex color.

    normalise:
      - 'sum': divide by sum(counts) to keep spectral shape only
      - 'max': divide by max(counts)
      - 'none': use raw counts (brightness will dominate)
    """
    wl = []
    vals = []
    for k, nm in CHANNEL_WAVELENGTHS_NM.items():
        if k in ch:
            wl.append(float(nm))
            vals.append(float(ch[k]))

    wl = np.array(wl)
    vals = np.array(vals)

    if normalise == 'sum':
        s = float(vals.sum())
        if s > 0:
            vals = vals / s
    elif normalise == 'max':
        m = float(vals.max())
        if m > 0:
            vals = vals / m

    sd = colour.SpectralDistribution(dict(zip(wl, vals)), name='sensor')

    # Resample to visible range; 5 nm is a good compromise for speed.
    shape = colour.SpectralShape(380, 780, 5)
    sd = sd.copy().align(shape)

    XYZ = colour.sd_to_XYZ(sd)  # default observer/illuminant settings
    rgb = colour.XYZ_to_sRGB(XYZ / 100)  # colour uses XYZ in [0,100]
    rgb = np.clip(rgb, 0, 1)

    return rgb01_to_hex(rgb.tolist())

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'message': 'Virtual Color Mixing Lab API is running'}), 200

@app.route('/api/create_session', methods=['POST'])
def create_session():
    try:
        data = request.json or {}
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400

        seed = int(data.get('random_seed', 42))
        sys_noise = float(data.get('systematic_noise_level', 0.10))
        rnd_noise = float(data.get('random_noise_level', 0.02))

        sessions[session_id] = VirtualColorMixingLab(random_seed=seed,
                                                    systematic_noise_level=sys_noise,
                                                    random_noise_level=rnd_noise)

        return jsonify({'status': 'created', 'session_id': session_id}), 201

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/execute_batch', methods=['POST'])
def execute_batch():
    try:
        data = request.json or {}
        session_id = data.get('session_id')
        code = data.get('code')
        if not session_id or session_id not in sessions:
            return jsonify({'error': 'Invalid session_id. Call /api/create_session first.'}), 400
        if not code:
            return jsonify({'error': 'No code provided'}), 400

        lab = sessions[session_id]

        safe_globals = {
            '__builtins__': __builtins__,
            'range': range,
            'len': len,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'zip': zip,
            'enumerate': enumerate,
            'print': print,
        }
        safe_locals = {}

        stdout = io.StringIO()
        import contextlib
        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, safe_globals, safe_locals)
        except Exception as exec_error:
            return jsonify({
                'error': 'Code execution error',
                'message': str(exec_error),
                'traceback': traceback.format_exc(),
                'stdout': stdout.getvalue()
            }), 400

        if 'experiment_design' not in safe_locals:
            return jsonify({'error': 'Code must define experiment_design', 'stdout': stdout.getvalue()}), 400

        experiment_design = safe_locals['experiment_design']
        if not isinstance(experiment_design, list):
            return jsonify({'error': 'experiment_design must be a list'}), 400

        df = lab.run_experiment_batch(experiment_design)
        results = df.to_dict(orient='records')

        # Add swatches (mixture from volumes, sensor from 8-channel spectrum)
        for r in results:
            R = float(r['Red']); Y = float(r['Yellow']); B = float(r['Blue'])
            tot = max(1e-9, R + Y + B)
            # Simple display mix: weighted average of three chosen primaries
            red_rgb = np.array([220, 60, 60], dtype=float)
            yellow_rgb = np.array([235, 200, 70], dtype=float)
            blue_rgb = np.array([60, 95, 215], dtype=float)
            mix = (R/tot)*red_rgb + (Y/tot)*yellow_rgb + (B/tot)*blue_rgb
            r['mixture_hex'] = f"#{int(mix[0]):02X}{int(mix[1]):02X}{int(mix[2]):02X}"

            ch = {k: r[k] for k in CHANNEL_WAVELENGTHS_NM.keys()}
            r['sensor_hex'] = sensor_channels_to_srgb_hex(ch, normalise='sum')
            r['sensor_intensity'] = float(sum(float(ch[k]) for k in ch))

        return jsonify({
            'status': 'success',
            'message': f'Executed {len(results)} experiments',
            'results': results,
            'stdout': stdout.getvalue()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/download_results', methods=['POST'])
def download_results():
    try:
        data = request.json or {}
        session_id = data.get('session_id')
        if not session_id or session_id not in sessions:
            return jsonify({'error': 'Invalid session_id'}), 400

        df = sessions[session_id].export_results_df()
        if df.empty:
            return jsonify({'error': 'No results to download'}), 400

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(io.BytesIO(output.getvalue().encode('utf-8')),
                         mimetype='text/csv',
                         as_attachment=True,
                         download_name=f'color_results_{session_id}.csv')

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

if __name__ == '__main__':
    print('Virtual Color Mixing Lab backend running at http://127.0.0.1:5000')
    app.run(debug=True, host='127.0.0.1', port=5000)
