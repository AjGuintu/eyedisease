import streamlit as st
from yolo_predictions import YOLO_Pred
from PIL import Image
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from reportlab.lib.utils import ImageReader
 
st.set_page_config(page_title="Cataract and Glaucoma Detection",
                   layout='wide',
                   page_icon='./images/object.png')
 
st.header('Get Object Detection for any Image')
st.write('Please Upload Image to get detections')
 
with st.spinner('Please wait while your model is loading'):
    yolo = YOLO_Pred(onnx_model='./models/best.onnx',
                    data_yaml='./models/data.yaml')
 
def upload_image():
    # Upload Image
    image_file = st.file_uploader(label='Upload Image')
    if image_file is not None:
        size_mb = image_file.size/(1024**2)
        file_details = {"filename":image_file.name,
                        "filetype":image_file.type,
                        "filesize": "{:,.2f} MB".format(size_mb)}
        # validate file
        if file_details['filetype'] in ('image/png','image/jpeg'):
            st.success('VALID IMAGE file type (png or jpeg)')
            return {"file":image_file,
                    "details":file_details}
        else:
            st.error('INVALID Image file type')
            st.error('Upload only png, jpg, jpeg')
            return None
 
def create_pdf_report(image_array, pred_img, custom_image_path="pages/Labels_Guide.jpg"):
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    
    # Calculate the center position for the image
    page_width, page_height = letter
    image_width = 300  # Adjust this based on your image size
    image_height = 300  # Adjust this based on your image size
    x_centered = (page_width - image_width) / 2
    y_centered = (page_height - image_height + 300) / 2
    
    # Convert and save the predicted image as bytes
    img_bytes = BytesIO()
    pred_img_obj = Image.fromarray(pred_img)
    pred_img_obj.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Draw the image centered on the PDF
    c.drawImage(ImageReader(img_bytes), x_centered, y_centered, width=image_width, height=image_height)
 
    # Load and draw the custom image (if provided)
    if custom_image_path:
        custom_img = Image.open(custom_image_path)
        custom_img_width = 500  # Adjust this based on your image size
        custom_img_height = 300  # Adjust this based on your image size
        x_custom = (page_width - custom_img_width) / 2
        y_custom = 50  # Adjust this to position the custom image vertically
        c.drawImage(ImageReader(custom_img), x_custom, y_custom, width=custom_img_width, height=custom_img_height)
    
    # Add informative text about labels
    labels_info2 = "Cataract and Glaucoma Report"
    c.setFont("Helvetica", 30)
    c.drawString(100, 750, labels_info2)
    labels_info3 = "RESULTS:"
    c.setFont("Helvetica", 20)
    c.drawString(110, 710, labels_info3)
    labels_info = "Green Labels: (Cataract), Blue Labels: (Glaucoma), Purple Labels: (Normal)"
    c.setFont("Helvetica", 12)
    c.drawString(120, 350, labels_info)
    
    c.showPage()
    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
def main():
    object = upload_image()
    
    if object:
        prediction = False
        image_obj = Image.open(object['file'])       
        
        col1 , col2 = st.columns(2)
        
        with col1:
            st.info('Preview of Image')
            st.image(image_obj)
            
        with col2:
            st.subheader('Check below for file details')
            st.json(object['details'])
            button = st.button('Get Detection from YOLO')
            if button:
                with st.spinner("""
                Getting Objects from image. Please wait...
                                """):
                    # Below command will convert
                    # object to array
                    image_array = np.array(image_obj)
                    pred_img = yolo.predictions(image_array)
                    prediction = True
                
        if prediction:
            st.subheader("Predicted Image")
            st.caption("Object detection from YOLO V5 model")
            st.image(Image.fromarray(pred_img))  # Display the detection image
            
            # Add a button to generate and download the PDF report with the predicted image
            
            pdf_data = create_pdf_report(image_array, pred_img)
            st.download_button(label="Generate and Download PDF Report", data=pdf_data, file_name="detection_report.pdf", mime='application/pdf')
 
if __name__ == "__main__":
    main()
 
