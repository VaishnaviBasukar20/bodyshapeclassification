import streamlit as st
from utils.utitlity import load_css, predictShape
from PIL import Image
import time

@st.experimental_dialog("Want to proceed further ?")
def confirmaton(test_img):
    cols = st.columns(6)
    if cols[0].button("Yes"):
        test_path = r"utils\testing.png"
        with st.status("Processing"):
            test_img.save(test_path)
            shape = predictShape(test_path)
            with open("result.txt","w") as file:
                file.write(shape)
        st.switch_page("pages/Result.py")
    if cols[1].button("No"):
        st.experimental_rerun()

def app(page="bodyScan"):
    st.set_page_config(page_icon="assests\logo.png",
                       page_title="Myntra", layout="wide", initial_sidebar_state="collapsed")
    load_css(f"style\{page}.css")

    with st.container():
        st.html("""<div id = 'main-container'></div>""")
        with st.container(border=False):
            st.html("""<div id = 'head-container'></div>""")
            cols = st.columns(2, gap="large", vertical_alignment="center")
            with cols[0]:
                sub_cols = st.columns(
                    [0.3, 1, 1, 1, 1], vertical_alignment='center')
                sub_cols[1].image("assests\Myntra-logo.png", width=250)
            with cols[1]:
                sub = st.columns([1, 1, 1, 0.5, 4, 1],
                                 vertical_alignment="center")
                sub[3].image("assests\search.png", width=20)
                sub[4].text_input(
                    "", label_visibility="collapsed", placeholder="search")

        with st.container(border=False, height=900):
            st.html("""<div id = 'inner-container'></div>""")
            st.html('''<h2 style="padding-left:50px">Scan the body :-</h2>''')
            cols = st.columns([1, 0.4, 0.5, 4, 1])
            cols[2].subheader("Select")
            selection = None
            with cols[3]:
                selection = st.selectbox(
                    "Use", ['Camera', 'Upload from the device'], label_visibility="collapsed", index=None)
                if selection:
                    process_image=None
                    if selection == "Camera":
                        cols=st.columns([20,200])
                        img = Image.open("assests\camera.png").resize((100, 100))
                        cols[0].image(img)
                        cam_img = st.camera_input(
                            "", label_visibility="collapsed")
                        process_image=cam_img
                    elif selection == "Upload from the device":
                        st.markdown("#### Upload")
                        img = st.file_uploader("file", label_visibility="collapsed", type=[
                                               "png", "jpeg", "jpg","webp"])
                        if img:
                            img = Image.open(img).resize((400, 400))
                            st.image(img)
                            process_image=img
                        else:
                            st.header("")
                            st.header("")
                            cols=st.columns(3)
                            img = Image.open(r"assests\upload.png").resize((300, 300))
                            cols[1].image(img)
                            
                    if process_image and st.button("Examine", use_container_width=True, type="primary"):
                        confirmaton(process_image)
            if not selection:
                st.header("")
                st.header("")
                cols=st.columns([1.5,1,1,1,1])
                img = Image.open("assests\camera.png").resize((300, 300))
                cols[1].image(img)
                img = Image.open("assests\cross.png").resize((300, 300))
                cols[2].image(img)
                img = Image.open(r"assests\upload.png").resize((300, 300))
                cols[3].image(img)
        st.header("")


if __name__ == "__main__":
    app()
