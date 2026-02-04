from ultralytics import YOLO

# Load the trained model weights
model = YOLO('runs/segment/train/weights/best.pt')

# Run inference on a specified image directory with various parameters
results = model.predict(
    source="",                                      # Images to be inferred (input directory)
    batch=2,                                        # Batch size for inference
    conf=0.25,                                      # Confidence threshold
    iou=0.6,                                        # IoU threshold for NMS
    imgsz=640,                                      # Input image size
    half=False,                                     # Use half-precision (FP16) inference
    # device=None,                                  # Inference device; None enables automatic selection (e.g., 'cpu', '0')
    max_det=300,                                    # Maximum number of detections per image
    vid_stride=1,                                  # Video frame stride
    stream_buffer=False,                            # Enable video stream buffering
    visualize=False,                                # Visualize intermediate model features
    augment=False,                                  # Enable test-time augmentation
    agnostic_nms=False,                             # Use class-agnostic NMS
    classes=None,                                   # Specify classes to detect (None = all classes)
    retina_masks=False,                             # Use high-resolution segmentation masks
    embed=None,                                     # Extract feature embeddings from a specific layer
    show=False,                                     # Display inference results
    save=True,                                      # Save inference outputs
    save_frames=False,                              # Save video frames as images
    save_txt=True,                                  # Save detection results to text files
    save_conf=False,                                # Save confidence scores to text files
    save_crop=False,                                # Save cropped detected objects
    show_labels=True,                               # Show class labels on detections
    show_conf=True,                                 # Show confidence scores on detections
    show_boxes=True,                                # Show bounding boxes
    line_width=None,                                # Line width of bounding boxes (e.g., 2, 4)
    device='0'                                      # Use GPU device 0
)
