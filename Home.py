import streamlit as st 

st.set_page_config(page_title="Home",
                   layout='wide',
                   page_icon='./images/home.png')

st.title("Cataract and Glaucoma Detection")
st.caption('This project demostrate Glaucoma and Cataract Detection through Retinal Images')

# Content
st.markdown("""
### Detect Glaucoma and Cataract
- Automatically detects  objects from image
- [Click here for App](/YOLO_for_image/)  

Below give are the object the our model will detect
1. Cataract
2. Normal
3. Glaucoma
           
            """)
