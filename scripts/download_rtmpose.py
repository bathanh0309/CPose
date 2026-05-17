import os
import urllib.request
import zipfile
from pathlib import Path

def download_and_extract():
    url = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-tiny_simcc-body7_pt-body7_420e-256x192-e872a938_20230504.zip"
    model_dir = Path("models/pose")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    final_model = model_dir / "rtmpose-t_256x192.onnx"
    
    if final_model.exists():
        print(f"[OK] model ready at {final_model}")
        return
        
    zip_path = model_dir / "rtmpose.zip"
    
    try:
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"[OK] downloaded")
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # find the .onnx file inside
            onnx_files = [f for f in zip_ref.namelist() if f.endswith('.onnx')]
            if not onnx_files:
                print("Error: No .onnx file found in zip")
                return
            
            # extract just that file
            extracted_path = zip_ref.extract(onnx_files[0], model_dir)
            
            # move to final model name
            os.rename(extracted_path, final_model)
            print(f"[OK] extracted")
            
    except Exception as e:
        print(f"Error downloading or extracting zip: {e}")
        print("Fallback to direct ONNX download...")
        direct_url = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.onnx"
        try:
            urllib.request.urlretrieve(direct_url, final_model)
            print(f"[OK] downloaded direct ONNX")
            print(f"[OK] extracted (not required)")
        except Exception as e2:
            print(f"Fallback failed: {e2}")
            return
            
    finally:
        if zip_path.exists():
            zip_path.unlink()
            
    print(f"[OK] model ready")

if __name__ == "__main__":
    download_and_extract()
