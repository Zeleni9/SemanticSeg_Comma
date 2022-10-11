import os
import cv2
import torch
from util import *
from LitModel import *
from argparse import ArgumentParser


# Setup device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    torch.device("cpu")
print("Device: ", device)


def inference_model(args):
    video_path = args.video_path
    model = LitModel.load_from_checkpoint(args.weights_path, **vars(args))
    model.eval()
    model.to(device)

    preprocess = get_preprocessing(model.preprocess_fn)
    transforms = get_valid_transforms(model.height, model.width)

    # Prepare video cap
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = cap.get(cv2.CAP_PROP_FPS)

    # # Create videoWriter to generate video from anonymized frames
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # video_writer = cv2.VideoWriter("lane_london.avi", fourcc, fps, (width, height), True)

    # Run inference on images from the video
    # counter = 0
    while cap.isOpened():
        ret, image = cap.read()
        if ret == True:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_vis = image.copy()
            image_vis = cv2.resize(image_vis, (args.width, args.height))

            sample = transforms(image=image)
            image = sample["image"]

            sample = preprocess(image=image)
            image = sample["image"]

            image = torch.from_numpy(image)  # Convert to tensor
            image = image.unsqueeze(0)  # Add batch dimension
            image = image.to(device)

            with torch.no_grad():
                output = model.forward(image)
                output = output.squeeze(0)
                output_predictions = output.argmax(0)

            output_predictions = output_predictions.cpu().numpy()
            prediction_colormap = decode_segmap(output_predictions, 5)
            overlay = get_overlay(image_vis, prediction_colormap, args.width, args.height)

            # overlay_vis = cv2.cvtColor(prediction_colormap, cv2.COLOR_BGR2RGB)
            # im_h = cv2.hconcat([cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB), overlay_vis])
            # cv2.imwrite(f"result/frame_{counter}.jpg", im_h)
            # video_writer.write(im_h)
            # counter += 1

            cv2.imshow("Source Image", cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB))
            cv2.imshow("Overlayed Image", cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            # cv2.imshow("Binary Image", cv2.cvtColor(prediction_colormap, cv2.COLOR_BGR2RGB))
            # (binary_seg_image[0] * 255).astype(np.uint8))

            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                break
        else:
            break
    cap.release()
    # video_writer.release()
    return


if __name__ == "__main__":
    """
    Inference on custom images
    Run with:
        python inference_demo.py --backbone efficientnet-b4 --height 874 --width 1164  --video_path .\center_video.avi --weights_path .\epoch.28_val_loss.0.0439.ckpt
    """
    root_dir = os.path.dirname(os.path.realpath(__file__))

    # Init arguments
    parent_parser = ArgumentParser(add_help=False)
    parser = LitModel.add_model_specific_args(parent_parser)
    parser.add_argument("--video_path", type=str, help="The video path or the src image save dir")
    parser.add_argument("--weights_path", type=str, help="Path to the checkpoint of weights")
    args = parser.parse_args()

    inference_model(args)
