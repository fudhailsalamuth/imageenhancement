import torch
# import torch.nn as nn # No longer needed for EDSR model definition
import torchvision.transforms as transforms # Used for converting tensor to PIL? Check usage
from pathlib import Path
import warnings
import sys
import requests
import os
import traceback
import tempfile
import functools
import time
# import sys # Already imported above
# print(sys.executable) # Keep for debugging if needed
# print(sys.path) # Keep for debugging if needed

# --- GUI ---
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, UnidentifiedImageError, ImageTk

# --- Dependency Check: Pillow(Tk), OpenCV, NumPy ---
try:
    _ = Image.new('RGB', (60, 30), color = 'red')
    from PIL import ImageTk # Check if ImageTk sub-module exists
    print("✅ Pillow and ImageTk found.")
except ImportError:
    print("ERROR: Missing 'Pillow' library or Tkinter support (ImageTk). Install: pip install Pillow")
    sys.exit(1)
try:
    import cv2
    import numpy as np
    # Check if dnn_superres is available
    _ = cv2.dnn_superres.DnnSuperResImpl_create()
    print(f"✅ OpenCV Version: {cv2.__version__} (with dnn_superres)")
    print(f"✅ NumPy Version: {np.__version__}")
except ImportError:
    print("ERROR: Missing 'opencv-python' or 'numpy'. Install: pip install opencv-python numpy")
    sys.exit(1)
except AttributeError:
    print("--------------------------------------------------------------------")
    print("ERROR: OpenCV installation does not include 'dnn_superres'.")
    print("Try installing 'opencv-contrib-python' instead of 'opencv-python':")
    print("   pip uninstall opencv-python")
    print("   pip install opencv-contrib-python")
    print("Or ensure you have a recent version of opencv-python.")
    print("--------------------------------------------------------------------")
    sys.exit(1)


# --- Dependency Check: DeOldify ---
try:
    from deoldify import device
    from deoldify.device_id import DeviceId
    from deoldify.visualize import get_image_colorizer, ModelImageVisualizer
    print("✅ DeOldify found.")
except ImportError as e:
    print("ERROR: Failed to import 'deoldify'. Install: pip install deoldify requests Pillow opencv-python numpy")
    print(f"Original error: {e}")
    sys.exit(1)

# --- REMOVED SuperImage Dependency Check ---


# --- Configuration Constants ---
MODELS_DIR = Path('./models') # For DeOldify AND OpenCV models now
ARTISTIC_MODEL_FILENAME = "ColorizeArtistic_gen.pth"
ARTISTIC_MODEL_URL = "https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth"
# OpenCV DNN SuperRes Model Config
OPENCV_UPSCALER_MODEL_NAME = "espcn" # Options: "espcn", "fsrcnn", "lapsrn", "edsr"
OPENCV_MODEL_URL_BASE = "https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/"
# GUI Config
MAX_PREVIEW_WIDTH = 400
MAX_PREVIEW_HEIGHT = 400

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning)

# --- Global Variables ---
# Models & Device
deoldify_colorizer = None
opencv_upscaler = None # Global variable for the OpenCV upscaler object
opencv_upscaler_scale = 0 # Track loaded scale
processing_device = None # 'cpu' or 'cuda'

# GUI Elements & Variables (same as before)
root = None
status_var = None
progress_var = None
input_path_var = None
output_path_var = None
scale_var = None
render_factor_var = None
denoise_var = None
sharpen_var = None
contrast_var = None
watermark_var = None
input_image_label = None
output_image_label = None
input_photo = None
output_photo = None

# --- Helper Function to Download Models ---
# Modified to handle different extensions like .pb
def download_model_if_needed(url: str, dest_path: Path, model_name: str):
    """Downloads a model file if it doesn't exist."""
    global status_var, root
    if dest_path.exists():
        print(f"✅ Model '{model_name}' ({dest_path.name}) found locally.")
        return True

    print(f"⏳ Model '{model_name}' ({dest_path.name}) not found. Downloading from {url}...")
    if status_var:
        status_var.set(f"Downloading {model_name}...")
        if root: root.update_idletasks()

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(url, stream=True, timeout=600) # Long timeout for potentially large models
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 # 1 MB chunks
        progress_dl = 0

        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
                progress_dl += len(chunk)
                if total_size > 0:
                     percent = (progress_dl / total_size) * 100
                     # Basic progress bar for console
                     sys.stdout.write(f"\rDownloading {model_name}: [{int(50 * progress_dl / total_size)*'='}>{(50-int(50 * progress_dl / total_size))*' '}] {percent:.1f}% ")
                     sys.stdout.flush()
                     # Update GUI status
                     if status_var: status_var.set(f"Downloading {model_name}: {percent:.1f}%")
                     if root: root.update_idletasks()
                else:
                    # Unknown size, just show downloaded amount
                    sys.stdout.write(f"\rDownloading {model_name}: {progress_dl/1024/1024:.1f} MB")
                    sys.stdout.flush()
                    if status_var: status_var.set(f"Downloading {model_name}: {progress_dl/1024/1024:.1f} MB")
                    if root: root.update_idletasks()


        print(f"\n✅ Download complete for {model_name}.")
        if status_var: status_var.set(f"{model_name} download complete.")
        if root: root.update_idletasks()
        return True

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error downloading {model_name} from {url}: {e}")
        if status_var: status_var.set(f"Error downloading {model_name}: {e}")
        # Clean up incomplete download
        if dest_path.exists():
            try: os.remove(dest_path)
            except OSError: pass
        return False
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during {model_name} download: {e}")
        if status_var: status_var.set(f"Error during {model_name} download: {e}")
        if dest_path.exists():
            try: os.remove(dest_path)
            except OSError: pass
        return False

# --- Core Image Colorization Function ---
# (Keep the original colorize_image function - returning PIL image)
def colorize_image(
    colorizer: ModelImageVisualizer,
    input_path: Path,
    output_path: Path,
    render_factor: int,
    watermark: bool
    ) -> Image.Image | None:
    """Colorizes a single image file using DeOldify. Returns PIL image on success."""
    global status_var, root
    if not input_path.is_file():
        print(f"❌ Error [Colorize]: Input image file not found: '{input_path}'")
        if status_var: status_var.set(f"Error: Input file not found {input_path.name}")
        return None

    print(f"\n🎨 Colorizing image: '{input_path}'...")
    print(f"   Render Factor: {render_factor}, Watermark: {watermark}")
    if status_var:
        status_var.set(f"Colorizing {input_path.name}...")
        if root: root.update_idletasks()

    try:
        start_time = time.time()
        output_image_pil = colorizer.get_transformed_image(
            path=input_path, render_factor=render_factor, watermarked=watermark
        )
        end_time = time.time()

        if output_image_pil is None:
            print("❌ Error [Colorize]: DeOldify processing returned None.")
            if status_var: status_var.set(f"Error: DeOldify failed on {input_path.name}")
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_image_pil.save(output_path)
        print(f"✅ Successfully saved colorized image to: '{output_path}' (took {end_time - start_time:.2f}s)")
        if status_var: status_var.set(f"Colorized {input_path.name}.")
        if root: root.update_idletasks()
        return output_image_pil # Return the PIL image

    except UnidentifiedImageError:
        print(f"❌ Error [Colorize]: Cannot identify image file: '{input_path}'.")
        if status_var: status_var.set(f"Error: Cannot identify {input_path.name}")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred during colorization: {e}")
        if status_var: status_var.set(f"Error during colorization: {e}")
        traceback.print_exc()
        return None


# --- Image Enhancement Function (Using OpenCV DNN SuperRes) ---
def enhance_image(
    input_path: Path,
    output_path: Path,
    scale_factor: float,             # Upscale factor (e.g., 4.0 for 4x).
    denoise_strength: int,
    enhance_contrast: bool,
    sharpen_amount: float,
    loaded_opencv_upscaler: cv2.dnn_superres.DnnSuperResImpl | None, # Pass the loaded OpenCV upscaler
    ) -> Image.Image | None: # Return PIL image on success
    """Applies enhancements (OpenCV DNN scaling, denoising, contrast, sharpening)."""
    global status_var, root
    if not input_path.is_file():
        print(f"❌ Error [Enhance]: Input image file not found: '{input_path}'")
        if status_var: status_var.set(f"Error: Enhance input not found {input_path.name}")
        return None

    print(f"\n✨ Enhancing image: '{input_path}'...")
    if status_var:
        status_var.set(f"Enhancing {input_path.name}...")
        if root: root.update_idletasks()

    try:
        start_time = time.time()
        # Load with OpenCV
        img_cv = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if img_cv is None:
            print(f"❌ Error [Enhance]: OpenCV could not load image: '{input_path}'.")
            if status_var: status_var.set(f"Error: Cannot load {input_path.name} for enhance")
            return None

        original_shape = img_cv.shape
        has_alpha = len(img_cv.shape) == 3 and img_cv.shape[2] == 4
        img_bgr = img_cv[:,:,:3] if has_alpha else img_cv # Work on BGR

        # --- OpenCV Pre-processing Steps ---
        # 1. Denoise (BEFORE scaling)
        if denoise_strength > 0:
            print(f"   Applying Denoising (strength: {denoise_strength})...")
            if status_var: status_var.set(f"Denoising...")
            if root: root.update_idletasks()
            h = float(denoise_strength)
            img_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, h, h, 7, 21)
            print("      Denoising done.")

        # --- OpenCV DNN Upscaling ---
        upscaled_bgr = None
        if scale_factor > 1.0 and loaded_opencv_upscaler is not None:
            target_scale = int(scale_factor)
            model_name = loaded_opencv_upscaler.getAlgorithm()
            print(f"   Applying OpenCV Upscaling ({model_name} x{target_scale})...")
            if status_var: status_var.set(f"Upscaling ({model_name} x{target_scale})...")
            if root: root.update_idletasks()

            try:
                upscaled_bgr = loaded_opencv_upscaler.upsample(img_bgr)
                print(f"      OpenCV DNN Resized from {img_bgr.shape[:2]} to {upscaled_bgr.shape[:2]}.")
                img_bgr = upscaled_bgr # Update img_bgr to the upscaled version
            except cv2.error as upscale_err:
                print(f"❌ Error during OpenCV DNN Upscaling: {upscale_err}")
                print(f"   Model: {model_name}, Scale: x{target_scale}, Input shape: {img_bgr.shape}")
                print(f"   Ensure the model file is valid and supports this scale.")
                traceback.print_exc()
                if status_var: status_var.set(f"Error during upscale: {upscale_err}")
                print("   ⚠️ Proceeding without upscaling due to error.")
                scale_factor = 1.0 # Treat as no upscale happened
            except Exception as upscale_general_err:
                print(f"❌ Unexpected error during OpenCV DNN Upscaling: {upscale_general_err}")
                traceback.print_exc()
                if status_var: status_var.set(f"Error during upscale: {upscale_general_err}")
                print("   ⚠️ Proceeding without upscaling due to error.")
                scale_factor = 1.0


        elif scale_factor > 1.0:
             print(f"   ⚠️ Warning: Scale factor > 1.0 but OpenCV Upscaler not loaded. Skipping upscale.")
             if status_var: status_var.set(f"Warning: Upscaler not ready, skipping upscale.")

        # --- OpenCV Post-processing Steps ---
        # 3. Enhance Contrast (CLAHE)
        if enhance_contrast:
            print("   Applying Contrast Enhancement (CLAHE)...")
            if status_var: status_var.set(f"Applying Contrast...")
            if root: root.update_idletasks()
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            print("      Contrast enhancement done.")

        # 4. Sharpen
        if sharpen_amount > 0.0:
            print(f"   Applying Sharpening (amount: {sharpen_amount:.2f})...")
            if status_var: status_var.set(f"Sharpening...")
            if root: root.update_idletasks()
            img_bgr_float = img_bgr.astype(np.float32)
            kernel = np.array([[0, -sharpen_amount, 0],
                               [-sharpen_amount, 1 + 4 * sharpen_amount, -sharpen_amount],
                               [0, -sharpen_amount, 0]], dtype=np.float32)
            sharpened_float = cv2.filter2D(img_bgr_float, -1, kernel)
            img_bgr = np.clip(sharpened_float, 0, 255).astype(np.uint8)
            print("      Sharpening done.")

        # 5. Reconstruct final image (handle alpha)
        final_img_cv = None
        if has_alpha:
            alpha_channel = img_cv[:,:,3]
            if scale_factor > 1.0: # Check if upscaling actually happened
                alpha_channel = cv2.resize(alpha_channel, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            final_img_cv = cv2.merge((img_bgr, alpha_channel))
        else:
            final_img_cv = img_bgr

        # 6. Save the final enhanced image using OpenCV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        params = []
        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        elif output_path.suffix.lower() == '.png':
             params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

        save_success = cv2.imwrite(str(output_path), final_img_cv, params)
        if not save_success:
            raise IOError(f"cv2.imwrite failed to save to {output_path}")

        end_time = time.time()
        print(f"✅ Successfully saved enhanced image to: '{output_path}' (took {end_time - start_time:.2f}s)")
        if status_var: status_var.set(f"Enhancement saved: {output_path.name}")
        if root: root.update_idletasks()

        # Convert final OpenCV image to PIL for return/display
        if has_alpha:
            final_pil_image = Image.fromarray(cv2.cvtColor(final_img_cv, cv2.COLOR_BGRA2RGBA))
        else:
            final_pil_image = Image.fromarray(cv2.cvtColor(final_img_cv, cv2.COLOR_BGR2RGB))
        return final_pil_image

    except cv2.error as e:
         print(f"❌ Error [Enhance]: OpenCV error: {e}")
         if status_var: status_var.set(f"Error during enhancement (OpenCV): {e}")
         traceback.print_exc()
         return None
    except Exception as e:
        print(f"❌ An unexpected error occurred during enhancement: {e}")
        if status_var: status_var.set(f"Error during enhancement: {e}")
        traceback.print_exc()
        return None


# --- Model Setup Functions ---
def setup_device(force_cpu=False):
    """Determines and sets the processing device (GPU or CPU) for PyTorch."""
    global processing_device, status_var
    # This primarily affects DeOldify, as OpenCV DNN runs on CPU by default
    # but can be configured for GPU backends (more complex setup usually needed)
    if processing_device and ((force_cpu and processing_device.type == 'cpu') or \
                               (not force_cpu and processing_device.type == 'cuda')):
        print(f"PyTorch device already set to: {processing_device}")
        return processing_device

    use_gpu = torch.cuda.is_available() and not force_cpu
    if use_gpu:
        processing_device = torch.device("cuda")
        print("✅ Setting PyTorch device: CUDA (GPU)")
        if status_var: status_var.set("Using device: GPU (PyTorch)")
        try:
            device.set(device=DeviceId.GPU0) # Set DeOldify device hint
            torch.backends.cudnn.benchmark = True
        except Exception as e:
             print(f"⚠️ Warning: Failed to set DeOldify GPU device hint: {e}")
    else:
        processing_device = torch.device("cpu")
        reason = "(Forced by user)" if force_cpu else "(GPU not available/detected for PyTorch)"
        print(f"✅ Setting PyTorch device: CPU {reason}")
        if status_var: status_var.set(f"Using device: CPU {reason}")
        try:
            device.set(device=DeviceId.CPU) # Set DeOldify device hint
        except Exception as e:
            print(f"⚠️ Warning: Failed to set DeOldify CPU device hint: {e}")

    print(f"   PyTorch operations will use: {processing_device}")
    print(f"   OpenCV DNN upscaling will use: CPU (default)") # Inform user
    return processing_device

def load_deoldify_model():
    """Loads the DeOldify Artistic model."""
    global deoldify_colorizer, status_var, root
    # (Function remains the same as previous version)
    if deoldify_colorizer:
        print("✅ DeOldify model already loaded.")
        return deoldify_colorizer

    model_path = MODELS_DIR / ARTISTIC_MODEL_FILENAME
    if not download_model_if_needed(ARTISTIC_MODEL_URL, model_path, "DeOldify Artistic"):
        if status_var: status_var.set("Failed to download DeOldify model.")
        return None

    original_torch_load = torch.load
    @functools.wraps(original_torch_load)
    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        if kwargs.get('weights_only') is True: print("⚠️ Warning: Overriding weights_only=True for DeOldify.")
        kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load
    print("\n⏳ Initializing DeOldify Artistic Colorizer...")
    if status_var:
        status_var.set("Loading DeOldify model...")
        if root: root.update_idletasks()
    try:
        (Path(".") / 'dummy_for_fastai').mkdir(exist_ok=True)
        deoldify_colorizer = get_image_colorizer(artistic=True, root_folder=Path("."))
        print("✅ DeOldify colorizer initialized successfully.")
        if status_var: status_var.set("DeOldify model loaded.")
        if root: root.update_idletasks()
        return deoldify_colorizer
    except Exception as e:
        print(f"\n❌ ERROR: Failed to initialize DeOldify colorizer: {e}")
        if status_var: status_var.set(f"Error loading DeOldify: {e}")
        traceback.print_exc()
        return None
    finally:
        if torch.load is not original_torch_load:
            torch.load = original_torch_load


# --- NEW: Function to Load OpenCV Upscaler Model ---
def load_opencv_upscaler_model(scale: int):
    """Loads the OpenCV DNN SuperRes model for the specified scale."""
    global opencv_upscaler, opencv_upscaler_scale, status_var, root

    if opencv_upscaler and opencv_upscaler_scale == scale:
        print(f"✅ OpenCV Upscaler ({OPENCV_UPSCALER_MODEL_NAME} x{scale}) already loaded.")
        return opencv_upscaler

    # Validate scale
    supported_scales = [2, 3, 4] # ESPCN supports these
    if scale not in supported_scales:
        print(f"❌ Error: Scale x{scale} not supported by {OPENCV_UPSCALER_MODEL_NAME} in this script.")
        if status_var: status_var.set(f"Error: Upscale x{scale} not supported.")
        return None

    model_file_name = f"{OPENCV_UPSCALER_MODEL_NAME.upper()}_x{scale}.pb" # e.g., ESPCN_x4.pb
    model_path = MODELS_DIR / model_file_name
    model_url = f"{OPENCV_MODEL_URL_BASE}{model_file_name}"
    model_label = f"OpenCV {OPENCV_UPSCALER_MODEL_NAME.upper()} x{scale}"

    # Download the model file if it doesn't exist
    if not download_model_if_needed(model_url, model_path, model_label):
        if status_var: status_var.set(f"Failed to download {model_label} model.")
        return None

    print(f"\n⏳ Loading {model_label} model from {model_path}...")
    if status_var:
        status_var.set(f"Loading {model_label} model...")
        if root: root.update_idletasks()

    try:
        # Create the DNN Super Resolution object
        sr = cv2.dnn_superres.DnnSuperResImpl_create()

        # Read the model
        sr.readModel(str(model_path))

        # Set the model name and scale
        # The model name must match the algorithm (e.g., 'espcn', 'edsr')
        # The scale must match the model file (e.g., 4 for ESPCN_x4.pb)
        sr.setModel(OPENCV_UPSCALER_MODEL_NAME, scale)

        # Store the loaded model and scale globally
        opencv_upscaler = sr
        opencv_upscaler_scale = scale
        print(f"✅ {model_label} model loaded successfully.")
        if status_var: status_var.set(f"{model_label} model loaded.")
        if root: root.update_idletasks()
        return opencv_upscaler

    except cv2.error as e:
        print(f"❌ Error loading OpenCV DNN model: {e}")
        print(f"   Model path: {model_path}")
        print(f"   Check if the file is valid and OpenCV has DNN support.")
        if status_var: status_var.set(f"Error loading OpenCV model: {e}")
        traceback.print_exc()
        opencv_upscaler = None
        opencv_upscaler_scale = 0
        return None
    except Exception as e:
        print(f"❌ Unexpected error loading OpenCV DNN model: {e}")
        if status_var: status_var.set(f"Error loading OpenCV model: {e}")
        traceback.print_exc()
        opencv_upscaler = None
        opencv_upscaler_scale = 0
        return None


# --- GUI Image Display Helper ---
# (Keep the original display_image_in_gui function)
def display_image_in_gui(image_source, target_label):
    """Loads, resizes, and displays an image in a Tkinter Label."""
    global input_photo, output_photo # Need to hold references

    if image_source is None:
        target_label.config(image='', text="Preview N/A")
        if target_label == input_image_label: input_photo = None
        elif target_label == output_image_label: output_photo = None
        return
    try:
        if isinstance(image_source, (str, Path)): image = Image.open(image_source)
        elif isinstance(image_source, Image.Image): image = image_source.copy()
        else:
             print(f"❌ Error [Display]: Unsupported image source type: {type(image_source)}")
             target_label.config(image='', text="Load Error")
             return

        img_w, img_h = image.size
        ratio = min(MAX_PREVIEW_WIDTH / img_w, MAX_PREVIEW_HEIGHT / img_h)
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)
        image.thumbnail((new_w, new_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        target_label.config(image=photo, text="")
        target_label.image = photo
        if target_label == input_image_label: input_photo = photo
        elif target_label == output_image_label: output_photo = photo
    except FileNotFoundError:
        print(f"❌ Error [Display]: Image file not found: {image_source}")
        target_label.config(image='', text="File Not Found")
    except UnidentifiedImageError:
        print(f"❌ Error [Display]: Cannot identify image file: {image_source}")
        target_label.config(image='', text="Invalid Image")
    except Exception as e:
        print(f"❌ Error [Display]: Failed to display image: {e}")
        target_label.config(image='', text="Display Error")
        traceback.print_exc()


# --- Main Processing Logic ---
def run_processing():
    """The main function called by the GUI button."""
    global deoldify_colorizer, opencv_upscaler, processing_device, status_var, progress_var, root
    global input_path_var, output_path_var, scale_var, render_factor_var
    global denoise_var, sharpen_var, contrast_var, watermark_var, output_image_label

    # (Input/Output path checks remain the same)
    input_path_str = input_path_var.get()
    output_dir_str = output_path_var.get()
    if not input_path_str or not output_dir_str:
        status_var.set("Error: Select Input Image AND Output Folder.")
        tk.messagebox.showerror("Error", "Please select both an input image and an output folder.")
        return
    input_path = Path(input_path_str)
    output_dir = Path(output_dir_str)
    if not input_path.is_file():
        status_var.set(f"Error: Input file not found: {input_path.name}")
        tk.messagebox.showerror("Error", f"Input image not found:\n{input_path_str}")
        return
    if not output_dir.is_dir():
        status_var.set(f"Error: Output folder not found: {output_dir.name}")
        tk.messagebox.showerror("Error", f"Output folder not found or is not a directory:\n{output_dir_str}")
        return

    # --- Get Parameters from GUI ---
    try:
        scale_str = scale_var.get()
        if scale_str == "None": scale_factor = 1.0
        elif scale_str == "x2": scale_factor = 2.0
        elif scale_str == "x3": scale_factor = 3.0
        elif scale_str == "x4": scale_factor = 4.0
        else: scale_factor = 1.0
        render_factor = int(render_factor_var.get())
        watermark = watermark_var.get()
        denoise_strength = int(denoise_var.get())
        enhance_contrast = contrast_var.get()
        sharpen_amount = sharpen_var.get()
        force_cpu = False # Affects PyTorch device mainly
    except ValueError as e:
        status_var.set("Error: Invalid parameter value.")
        tk.messagebox.showerror("Error", f"Invalid parameter value entered:\n{e}")
        return

    # (Parameter validation remains the same)
    if not (7 <= render_factor <= 40):
        status_var.set(f"Error: Render factor ({render_factor}) must be 7-40.")
        tk.messagebox.showerror("Error", f"Render factor must be between 7 and 40.")
        return
    if denoise_strength < 0: denoise_strength = 0
    if sharpen_amount < 0.0: sharpen_amount = 0.0

    # (Print starting message remains the same)
    print("\n--- Starting Processing ---")
    print(f"Input: {input_path}")
    print(f"Output Dir: {output_dir}")
    print(f"Parameters:")
    print(f"  Scale: {'None' if scale_factor == 1.0 else f'x{int(scale_factor)}'} (Using OpenCV {OPENCV_UPSCALER_MODEL_NAME.upper()})")
    print(f"  Render Factor: {render_factor}")
    # ... rest of parameter printing ...

    display_image_in_gui(None, output_image_label) # Clear output preview

    enhancements_enabled = (scale_factor > 1.0 or denoise_strength > 0 or enhance_contrast or sharpen_amount > 0.0)

    # 1. Setup PyTorch Device (for DeOldify)
    if not processing_device: setup_device(force_cpu)
    if not processing_device:
         status_var.set("Error: Could not set up PyTorch processing device.")
         tk.messagebox.showerror("Error", "Could not set up PyTorch processing device (CPU/GPU). Check logs.")
         return

    # 2. Load DeOldify Model
    deoldify_colorizer = load_deoldify_model()
    if not deoldify_colorizer:
        status_var.set("Error: Failed to load DeOldify model.")
        tk.messagebox.showerror("Error", "Failed to load DeOldify colorization model. Check logs.")
        return

    # 3. Load OpenCV Upscaler Model if needed
    loaded_cv_upscaler = None
    if scale_factor > 1.0:
        loaded_cv_upscaler = load_opencv_upscaler_model(scale=int(scale_factor))
        if not loaded_cv_upscaler:
            print(f"⚠️ OpenCV {OPENCV_UPSCALER_MODEL_NAME.upper()} model failed to load. Proceeding without upscaling.")
            status_var.set("Warning: Upscaler failed. Skipping upscale.")
            scale_factor = 1.0 # Disable scaling for this run
            enhancements_enabled = (denoise_strength > 0 or enhance_contrast or sharpen_amount > 0.0)
            tk.messagebox.showwarning("Warning", f"OpenCV {OPENCV_UPSCALER_MODEL_NAME.upper()} upscaling model failed to load. Proceeding without upscaling.")

    # (Reset progress bar remains the same)
    if progress_var: progress_var.set(0)
    if root: root.update_idletasks()

    # 4. Determine Output Paths (remains the same, uses scale_factor)
    base_filename = input_path.stem
    colorized_suffix = "_colorized"
    enhanced_suffix = "_enhanced"
    # Use loaded_cv_upscaler to check if scaling is actually possible for filename
    scale_suffix = f"_x{int(scale_factor)}" if scale_factor > 1.0 and loaded_cv_upscaler else ""

    intermediate_colorized_file_to_delete = None
    final_output_path = None
    colorized_output_path = None
    processed_pil_image = None

    if not enhancements_enabled:
         final_output_path = output_dir / f"{base_filename}{colorized_suffix}.png"
         colorized_output_path = final_output_path
    else:
        try:
            with tempfile.NamedTemporaryFile(delete=False, dir=output_dir,
                                             prefix=f"{base_filename}_temp_colorized_",
                                             suffix=".png") as temp_f:
                colorized_output_path = Path(temp_f.name)
            intermediate_colorized_file_to_delete = colorized_output_path
            print(f"   Intermediate colorized file will be: {colorized_output_path}")
        except Exception as temp_err:
             print(f"❌ Error creating temporary file: {temp_err}. Cannot proceed safely.")
             status_var.set("Error creating temporary file.")
             tk.messagebox.showerror("Error", f"Failed to create temporary file in output directory:\n{temp_err}")
             return
        final_output_path = output_dir / f"{base_filename}{colorized_suffix}{scale_suffix}{enhanced_suffix}.png"


    # 5. Perform Colorization
    status_var.set(f"Starting colorization...")
    if root: root.update_idletasks()
    colorized_pil_image = colorize_image(
        colorizer=deoldify_colorizer,
        input_path=input_path,
        output_path=colorized_output_path,
        render_factor=render_factor,
        watermark=watermark
    )
    colorization_success = colorized_pil_image is not None
    if not colorization_success:
        # (Error handling same as before)
        status_var.set("Colorization failed. Check logs.")
        tk.messagebox.showerror("Error", "Colorization step failed. Check console logs for details.")
        if intermediate_colorized_file_to_delete and intermediate_colorized_file_to_delete.is_file():
             try: os.remove(intermediate_colorized_file_to_delete)
             except OSError: pass
        return
    if progress_var: progress_var.set(50)
    if root: root.update_idletasks()
    processed_pil_image = colorized_pil_image


    # 6. Perform Enhancement
    enhancement_success = True
    if enhancements_enabled:
        status_var.set(f"Starting enhancement...")
        if root: root.update_idletasks()
        enhanced_pil_image = enhance_image( # Call the updated enhance_image
            input_path=colorized_output_path,  # Input is the intermediate colorized file
            output_path=final_output_path,     # Output is the final desired file path
            scale_factor=scale_factor,
            denoise_strength=denoise_strength,
            enhance_contrast=enhance_contrast,
            sharpen_amount=sharpen_amount,
            loaded_opencv_upscaler=loaded_cv_upscaler # Pass the OpenCV upscaler
        )
        enhancement_success = enhanced_pil_image is not None
        if enhancement_success:
            processed_pil_image = enhanced_pil_image
        else:
            # (Error handling same as before)
             status_var.set("Enhancement failed. Check logs.")
             tk.messagebox.showerror("Error", "Enhancement step failed. Check console logs for details.")
             final_output_path = colorized_output_path

    elif not enhancements_enabled:
         # (Message same as before)
         print("   No enhancements requested, using colorized result.")
         status_var.set("Colorization complete (no enhancements).")


    # 7. Cleanup Temporary Intermediate File
    # (Logic remains the same)
    if intermediate_colorized_file_to_delete and intermediate_colorized_file_to_delete.is_file():
        if enhancement_success or not enhancements_enabled:
            try:
                print(f"🧹 Cleaning up intermediate file: {intermediate_colorized_file_to_delete}")
                os.remove(intermediate_colorized_file_to_delete)
            except OSError as e:
                print(f"⚠️ Warning: Could not delete intermediate file {intermediate_colorized_file_to_delete}: {e}")
        else:
            print(f"   Keeping intermediate colorized file due to enhancement failure: {intermediate_colorized_file_to_delete}")


    if progress_var: progress_var.set(100)

    # 8. Final Status & Display Output
    # (Logic remains the same)
    overall_success = colorization_success and enhancement_success
    if overall_success:
        final_msg = f"Success! Output: {final_output_path.name}"
        print(f"\n🎉 Image processing finished successfully. Final output: '{final_output_path}'")
        status_var.set(final_msg)
        display_image_in_gui(processed_pil_image, output_image_label)
    else:
        if colorization_success and not enhancement_success:
            final_msg = "Enhancement failed. Colorized result saved."
            status_var.set(final_msg)
            display_image_in_gui(processed_pil_image, output_image_label)
        else:
            final_msg = "Processing failed. Check console logs."
            status_var.set(final_msg)
            display_image_in_gui(None, output_image_label)
        print("\n⚠️ Image processing encountered errors.")


# --- Tkinter GUI Setup ---
# (select_input_file and select_output_folder remain the same)
def select_input_file():
    """Opens file dialog, sets input path, and displays preview."""
    global input_path_var, input_image_label, output_image_label
    filetypes = (("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*"))
    filename = filedialog.askopenfilename(title="Select Input Image", filetypes=filetypes)
    if filename:
        input_path_var.set(filename)
        print(f"Input selected: {filename}")
        display_image_in_gui(filename, input_image_label)
        display_image_in_gui(None, output_image_label)
        status_var.set("Input selected. Ready for processing.")

def select_output_folder():
    """Opens directory dialog to select output folder."""
    global output_path_var
    directory = filedialog.askdirectory(title="Select Output Folder")
    if directory:
        output_path_var.set(directory)
        print(f"Output folder selected: {directory}")

def build_gui():
    """Creates and runs the Tkinter GUI."""
    # (Global declarations remain the same)
    global status_var, progress_var, root, input_path_var, output_path_var
    global scale_var, render_factor_var, denoise_var, sharpen_var, contrast_var, watermark_var
    global input_image_label, output_image_label

    root = tk.Tk()
    root.title(f"Image Colorizer & Enhancer (DeOldify + OpenCV {OPENCV_UPSCALER_MODEL_NAME.upper()})") # Updated title
    root.minsize(800, 600)

    style = ttk.Style()
    style.theme_use('clam')

    # --- Variables (with default values) ---
    # (Remain the same)
    input_path_var = tk.StringVar()
    output_path_var = tk.StringVar()
    status_var = tk.StringVar(value="Ready. Select input image and output folder.")
    progress_var = tk.DoubleVar(value=0.0)
    scale_var = tk.StringVar(value="x4") # Default scale
    render_factor_var = tk.DoubleVar(value=35)
    denoise_var = tk.DoubleVar(value=0)
    sharpen_var = tk.DoubleVar(value=0.0)
    contrast_var = tk.BooleanVar(value=False)
    watermark_var = tk.BooleanVar(value=False)

    # --- GUI Layout (remains largely the same) ---
    main_frame = ttk.Frame(root, padding="10 10 10 10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    main_frame.columnconfigure(0, weight=1) # IO
    main_frame.columnconfigure(1, weight=3) # Params
    main_frame.columnconfigure(2, weight=3) # Input Preview
    main_frame.columnconfigure(3, weight=3) # Output Preview

    # --- IO Frame (remains the same) ---
    io_frame = ttk.LabelFrame(main_frame, text="Input / Output", padding="10")
    io_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=5, pady=5)
    io_frame.columnconfigure(0, weight=1)
    ttk.Label(io_frame, text="Input Image:").grid(row=0, column=0, sticky=tk.W, pady=(0,2))
    input_entry = ttk.Entry(io_frame, textvariable=input_path_var, width=30)
    input_entry.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)
    input_button = ttk.Button(io_frame, text="Browse...", command=select_input_file)
    input_button.grid(row=1, column=2, sticky=tk.E, padx=(5, 0), pady=2)
    ttk.Label(io_frame, text="Output Folder:").grid(row=2, column=0, sticky=tk.W, pady=(5,2))
    output_entry = ttk.Entry(io_frame, textvariable=output_path_var, width=30)
    output_entry.grid(row=3, column=0, columnspan=2, sticky="ew", pady=2)
    output_button = ttk.Button(io_frame, text="Browse...", command=select_output_folder)
    output_button.grid(row=3, column=2, sticky=tk.E, padx=(5, 0), pady=2)

    # --- Parameter Frame (remains the same) ---
    param_frame = ttk.LabelFrame(main_frame, text="Processing Parameters", padding="10")
    param_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
    param_frame.columnconfigure(1, weight=1)
    # Upscale
    ttk.Label(param_frame, text="Upscale:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
    scale_combo = ttk.Combobox(param_frame, textvariable=scale_var,
                               values=["None", "x2", "x3", "x4"], state="readonly", width=8) # ESPCN supports these
    scale_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=3)
    # Render Factor
    ttk.Label(param_frame, text="Color Render Factor (7-40):").grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10,0))
    render_scale = ttk.Scale(param_frame, variable=render_factor_var, from_=7, to=40, orient=tk.HORIZONTAL, length=200)
    render_scale.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=1)
    render_label = ttk.Label(param_frame, text=f"{render_factor_var.get():.0f}")
    render_factor_var.trace_add("write", lambda *args: render_label.config(text=f"{render_factor_var.get():.0f}"))
    render_label.grid(row=2, column=2, sticky=tk.W, padx=5)
    # Denoise
    ttk.Label(param_frame, text="Denoise Strength (0-30):").grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10,0))
    denoise_scale = ttk.Scale(param_frame, variable=denoise_var, from_=0, to=30, orient=tk.HORIZONTAL, length=200)
    denoise_scale.grid(row=4, column=0, columnspan=2, sticky="ew", padx=5, pady=1)
    denoise_label = ttk.Label(param_frame, text=f"{denoise_var.get():.0f}")
    denoise_var.trace_add("write", lambda *args: denoise_label.config(text=f"{denoise_var.get():.0f}"))
    denoise_label.grid(row=4, column=2, sticky=tk.W, padx=5)
    # Sharpen
    ttk.Label(param_frame, text="Sharpen Amount (0.0-2.0):").grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(10,0))
    sharpen_scale = ttk.Scale(param_frame, variable=sharpen_var, from_=0.0, to=2.0, orient=tk.HORIZONTAL, length=200)
    sharpen_scale.grid(row=6, column=0, columnspan=2, sticky="ew", padx=5, pady=1)
    sharpen_label = ttk.Label(param_frame, text=f"{sharpen_var.get():.1f}")
    sharpen_var.trace_add("write", lambda *args: sharpen_label.config(text=f"{sharpen_var.get():.1f}"))
    sharpen_label.grid(row=6, column=2, sticky=tk.W, padx=5)
    # Checkboxes
    contrast_check = ttk.Checkbutton(param_frame, text="Enhance Contrast (CLAHE)", variable=contrast_var)
    contrast_check.grid(row=7, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(10, 2))
    watermark_check = ttk.Checkbutton(param_frame, text="Add DeOldify Watermark", variable=watermark_var)
    watermark_check.grid(row=8, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

    # --- Image Preview Frames (remain the same) ---
    preview_frame_input = ttk.LabelFrame(main_frame, text="Input Preview", padding="5")
    preview_frame_input.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=5, pady=5)
    preview_frame_input.rowconfigure(0, weight=1)
    preview_frame_input.columnconfigure(0, weight=1)
    input_image_label = ttk.Label(preview_frame_input, text="Select an input image", anchor="center", background='gray85', borderwidth=2, relief="sunken")
    input_image_label.grid(row=0, column=0, sticky="nsew")

    preview_frame_output = ttk.LabelFrame(main_frame, text="Output Preview", padding="5")
    preview_frame_output.grid(row=0, column=3, rowspan=2, sticky="nsew", padx=5, pady=5)
    preview_frame_output.rowconfigure(0, weight=1)
    preview_frame_output.columnconfigure(0, weight=1)
    output_image_label = ttk.Label(preview_frame_output, text="Output will appear here", anchor="center", background='gray85', borderwidth=2, relief="sunken")
    output_image_label.grid(row=0, column=0, sticky="nsew")

    # --- Bottom Panel (remains the same) ---
    run_button = ttk.Button(main_frame, text="Run Processing", command=run_processing)
    run_button.grid(row=2, column=0, columnspan=2, pady=10, padx=5, sticky="ew")
    progress_bar = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, length=400, mode='determinate', variable=progress_var)
    progress_bar.grid(row=3, column=0, columnspan=4, sticky="ew", padx=5, pady=(5,0))
    status_label = ttk.Label(main_frame, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
    status_label.grid(row=4, column=0, columnspan=4, sticky="ew", padx=5, pady=(2,5))

    # Configure row weights
    main_frame.rowconfigure(0, weight=0) # Params row should not expand vertically much
    main_frame.rowconfigure(1, weight=1) # This row contains the previews via rowspan, let it expand
    main_frame.rowconfigure(2, weight=0)
    main_frame.rowconfigure(3, weight=0)
    main_frame.rowconfigure(4, weight=0)

    root.mainloop()


if __name__ == "__main__":
    print(f"--- DeOldify + OpenCV {OPENCV_UPSCALER_MODEL_NAME.upper()} Image Processing Script ---")
    try: print(f"✅ PyTorch Version: {torch.__version__}")
    except NameError: print("❌ Error: PyTorch not found."); sys.exit(1)

    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✅ Models directory checked/created: {MODELS_DIR.resolve()}")
    except Exception as e:
        print(f"❌ Error creating models directory '{MODELS_DIR}': {e}")

    build_gui()