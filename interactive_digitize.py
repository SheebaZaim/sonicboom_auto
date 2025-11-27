# # interactive_digitize.py

# import os
# import argparse
# import numpy as np
# import pandas as pd
# import cv2
# import matplotlib.pyplot as plt
# from skimage import color, filters, morphology

# def extract_curve_with_clicks(image_path, output_path, n_samples=200000):
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"Could not load image: {image_path}")

#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     print(f"\nProcessing: {os.path.basename(image_path)}")
#     plt.imshow(img_rgb)
#     plt.title("1 & 2: Click X-axis (left then right)\n3 & 4: Click Y-axis (bottom then top)")
#     points = plt.ginput(4)
#     plt.close()

#     if len(points) < 4:
#         raise RuntimeError("Not enough points selected")

#     x1, y1 = points[0]
#     x2, y2 = points[1]
#     xt1, yt1 = points[2]
#     xt2, yt2 = points[3]

#     axis_x_min = float(input("Enter X-axis MIN value (from paper): "))
#     axis_x_max = float(input("Enter X-axis MAX value (from paper): "))
#     axis_y_min = float(input("Enter Y-axis MIN value (from paper): "))
#     axis_y_max = float(input("Enter Y-axis MAX value (from paper): "))

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5,5), 0)
#     th = filters.threshold_otsu(blur)
#     bw = blur < th  
#     bw = morphology.remove_small_objects(bw, min_size=100)
#     contours, _ = cv2.findContours(bw.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#     if not contours:
#         raise RuntimeError("Curve not detected!")

#     contour = max(contours, key=lambda c: len(c)).squeeze()

#     px = contour[:,0]; py = contour[:,1]
#     width = img.shape[1]; height = img.shape[0]

#     t_min_pix, p_max_pix = x1, yt2
#     t_max_pix, p_min_pix = x2, yt1

#     t = axis_x_min + (px - t_min_pix) / (t_max_pix - t_min_pix) * (axis_x_max - axis_x_min)

#     p = axis_y_max - (py - p_max_pix) / (p_min_pix - p_max_pix) * (axis_y_max - axis_y_min)

#     order = np.argsort(t)
#     t = t[order]; p = p[order]

#     t_uniform = np.linspace(t.min(), t.max(), n_samples)
#     p_uniform = np.interp(t_uniform, t, p)

#     df = pd.DataFrame({'t': t_uniform, 'p': p_uniform})
#     df.to_csv(output_path, index=False)
#     print(f"✔ Saved CSV: {output_path}")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--folder", required=True, help="Folder containing images")
#     parser.add_argument("--output_dir", default="outputs", help="Where to save CSVs")
#     parser.add_argument("--samples", type=int, default=200000, help="Number of samples")
#     args = parser.parse_args()

#     os.makedirs(args.output_dir, exist_ok=True)

#     image_files = [f for f in os.listdir(args.folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
#     if not image_files:
#         print("❌ No image found!")
#         return

#     for image in image_files:
#         img_path = os.path.join(args.folder, image)
#         csv_path = os.path.join(args.output_dir, f"{os.path.splitext(image)[0]}.csv")
#         extract_curve_with_clicks(img_path, csv_path, args.samples)

#     print("\n🎉 All images processed successfully!")
#     print(f"CSV files saved to {args.output_dir}/")

# if __name__ == '__main__':
#     main()

# interactive_digitize_auto.py
import os
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage import filters, morphology

def extract_curve_auto(image_path, output_path, n_samples=200000):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"\nProcessing: {os.path.basename(image_path)}")
    plt.imshow(img_rgb)
    plt.title("Click 4 points:\n1: X-min (left) 2: X-max (right)\n3: Y-min (bottom) 4: Y-max (top)")
    points = plt.ginput(4)
    plt.close()

    if len(points) < 4:
        raise RuntimeError("❌ Not enough points selected. Click 4 points.")

    # Extract clicked points
    x1, y1 = points[0]  # X-min
    x2, y2 = points[1]  # X-max
    xt1, yt1 = points[2]  # Y-min
    xt2, yt2 = points[3]  # Y-max

    # Preprocess image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = filters.threshold_otsu(blur)
    bw = blur < th
    bw = morphology.remove_small_objects(bw, min_size=100)
    contours, _ = cv2.findContours(bw.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise RuntimeError("❌ Curve not detected!")

    contour = max(contours, key=lambda c: len(c)).squeeze()
    px = contour[:, 0]
    py = contour[:, 1]

    # Automatic scaling using clicked points
    t_min_pix, t_max_pix = x1, x2
    p_min_pix, p_max_pix = yt1, yt2  # Y in pixels (top-left origin)

    # Map pixels to real coordinates: assume X goes from 0 to 1, Y from 0 to 1
    t = (px - t_min_pix) / (t_max_pix - t_min_pix)  # X normalized
    p = 1 - (py - p_min_pix) / (p_max_pix - p_min_pix)  # Y normalized (invert Y)

    # Interpolate to uniform number of points
    order = np.argsort(t)
    t = t[order]
    p = p[order]
    t_uniform = np.linspace(t.min(), t.max(), n_samples)
    p_uniform = np.interp(t_uniform, t, p)

    # Save CSV
    df = pd.DataFrame({'t': t_uniform, 'p': p_uniform})
    df.to_csv(output_path, index=False)
    print(f"✔ Saved CSV: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Folder containing images")
    parser.add_argument("--output_dir", default="outputs", help="Folder to save CSVs")
    parser.add_argument("--samples", type=int, default=200000, help="Number of samples")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(args.folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not image_files:
        print("❌ No image found!")
        return

    for image in image_files:
        img_path = os.path.join(args.folder, image)
        csv_path = os.path.join(args.output_dir, f"{os.path.splitext(image)[0]}.csv")
        try:
            extract_curve_auto(img_path, csv_path, args.samples)
        except Exception as e:
            print(f"⚠ Error processing {image}: {e}")

    print("\n🎉 All images processed successfully!")
    print(f"CSV files saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
