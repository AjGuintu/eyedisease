import threading
from typing import Union
import av
import numpy as np
import streamlit as st
import cv2
import os
from PIL import Image
from io import BytesIO
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

def main():
    class VideoTransformer(VideoTransformerBase):
        frame_lock: threading.Lock
        out_image: Union[np.ndarray, None]

        def __init__(self) -> None:
            self.frame_lock = threading.Lock()
            self.out_image = None

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            out_image = frame.to_ndarray(format="bgr24")
            with self.frame_lock:
                self.out_image = out_image
            return out_image

    ctx = webrtc_streamer(key="snapshot", video_transformer_factory=VideoTransformer)

    if ctx.video_transformer:
        snap = st.button("Snapshot")
        if snap:
            with ctx.video_transformer.frame_lock:
                out_image = ctx.video_transformer.out_image

            if out_image is not None:
                # Convert BGR to RGB for display
                st.write("Output image:")
                st.image(cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB), channels="RGB", width=500)

                # Ensure the directory exists
                data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Data/"))
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)

                # Save the image in RGB format
                out_image_rgb = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(data_dir, "filename.jpg"), out_image_rgb)
            else:
                st.warning("No frames available yet.")

        # Add a button to download the captured image
        if ctx.video_transformer.out_image is not None:
            with ctx.video_transformer.frame_lock:
                out_image = ctx.video_transformer.out_image

            out_image_rgb = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)

            # Convert the image to bytes for download
            image_pil = Image.fromarray(out_image_rgb)
            byte_io = BytesIO()
            image_pil.save(byte_io, format='JPEG')
            image_bytes = byte_io.getvalue()

            # Wrap the download button in a container to set width and height
            container = st.container()
            with container:
                st.download_button(
                    label="Download Image",
                    data=image_bytes,
                    file_name="downloaded_image.jpg",
                    mime="image/jpeg"
                )

if __name__ == "__main__":
    main()
