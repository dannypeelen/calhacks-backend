from inference import get_model

model = get_model("weapon-qpfo8/1")

image_src = "" #TODO: add path here

results = model.infer(image_src)


#This will VISUALIZE the results, if needed
#==================================================================

# import cv2
# import supervision as sv

# # Load the image for visualization (if not already loaded)
# image_np = cv2.imread(image_source)

# # Convert results to Supervision Detections format
# detections = sv.Detections.from_inference(results)

# # Create a bounding box annotator
# box_annotator = sv.BoxAnnotator()

# # Annotate the image
# annotated_image = box_annotator.annotate(scene=image_np.copy(), detections=detections)

# # Display or save the annotated image
# cv2.imshow("Inference Results", annotated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()