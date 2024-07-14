import streamlit as st
from utils.utitlity import load_css
from PIL import Image


def app(page="home"):
    """
    The function `app` sets up a web page for a Myntra application with specific configurations and
    layout, including elements like images, text inputs, buttons, and page switching options.
    
    :param page: The `page` parameter in the `app` function is used to specify the page to be displayed. The default value is set to "home", defaults to home (optional)
    """
    st.set_page_config(page_icon="assests\logo.png", page_title="Myntra",
                       layout="wide", initial_sidebar_state="collapsed")
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

        st.html('''<h2 class="bodyScanner" ><u>Body Shape Examiner</u></h2>''')

        with st.container(border=False):
            st.html("""<div id = 'inner-container'></div>""")
            cols = st.columns([1.5, 4, 4, 1], vertical_alignment="center")
            with cols[1]:
                img = Image.open(r"assests\body-scan.png").resize((450, 700))
                st.image(img)
            with cols[2]:
                img = Image.open(
                    r"assests\Body shape chart.jpg").resize((700, 650))
                st.image(img)
                sub_cols = st.columns(2)
                if sub_cols[0].button("Using body scan", type="primary", use_container_width=True):
                    st.switch_page("pages/BodyScan.py")
                if sub_cols[1].button("Using measurements", type="primary", use_container_width=True):
                    st.switch_page("pages/Measurements.py")


if __name__ == "__main__":
    app()
