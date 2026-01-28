# app.py
import os
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from model_utils import load_model, run_inference_on_image

# --------- CONFIG ----------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "static", "results")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.keras")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULTS_FOLDER"] = RESULTS_FOLDER

# Load model once at startup
model, last_conv_layer = load_model(MODEL_PATH)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(upload_path)

            # result subfolder (overwrite same "current" each time)
            result_subdir = os.path.join(app.config["RESULTS_FOLDER"], "current")
            os.makedirs(result_subdir, exist_ok=True)

            # run model inference + XAI
            results = run_inference_on_image(
                model,
                last_conv_layer,
                upload_path,
                out_dir=result_subdir,
                threshold=0.5
            )

            # convert absolute paths -> URLs
            def to_url(path):
                rel = os.path.relpath(path, BASE_DIR).replace("\\", "/")
                return "/" + rel

            uploaded_url = "/" + os.path.relpath(upload_path, BASE_DIR).replace("\\", "/")
            original_url = to_url(results["original"])
            mask_url = to_url(results["mask"])
            boxes_url = to_url(results["boxes"])
            gradcam_url = to_url(results["gradcam"])

            return render_template(
                "index.html",
                uploaded_image=uploaded_url,
                original_image=original_url,
                mask_image=mask_url,
                boxes_image=boxes_url,
                gradcam_image=gradcam_url,
                teeth_total=results["teeth_total"],
                cavity_teeth=results["cavity_teeth"],
                rct_teeth=results["rct_teeth"],
                show_results=True
            )

    # GET request: just show upload form
    return render_template("index.html", show_results=False)


if __name__ == "__main__":
    app.run(debug=True)
