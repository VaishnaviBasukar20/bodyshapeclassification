import streamlit as st
from utils.utitlity import load_css
from PIL import Image
import time

@st.experimental_dialog("Wait for few seconds")
def confirmaton():
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1)
    st.switch_page(page="pages/Recommendation.py")


def app(page="result"):
    with open("result.txt","r") as file:
        shape = file.read().strip().lower()
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
        
        st.html('''<h2 class="result" ><u><br>Your Body Shape</u></h2>''')

        with st.container(border=False,height=850) as c:
            st.html("""<div id = 'inner-container'></div>""")
            st.html(f'''<h3 style="text-align: center" >{shape.capitalize()}</h3>''')
            st.image(f"assests\{shape}\{shape}.webp", width=350)
            if shape=="pear":
                st.markdown('''**Aka**: Spoon, Triangle, A-Frame, Gynoid shape''')
                st.markdown("#### Upper body is smaller than the lower body")
                st.markdown("#### Shoulders are narrower than hips")
                st.markdown("#### Waist is fairly defined")
            elif shape=="rectangle":
                st.markdown('''**Aka**: Ruler, Banana, Straight, H-Frame, Athletic''')
                st.markdown("#### Body equally proportioned throughout")
                st.markdown("#### Shoulders, bust and hips are the approximately the same size")
            elif shape=="apple":
                st.markdown('''**Aka**: Diamond, Round, O-Frame, Android shape''')
                st.markdown("#### Waist is larger than hips")
                st.markdown("#### Shoulders and hips approximately the same size")
                st.markdown("#### Widest around the middle")
            elif shape=="hourglass":
                st.markdown('''**Aka**: Figure 8, X-Frame''')
                st.markdown("#### Upper and lower body are equally proportioned")
                st.markdown("#### Shoulders same (or almost the same) size as the hips")
                st.markdown("#### Clearly defined and pronounced waistline")
            elif shape=="inverted triangle":
                st.markdown('''**Aka**: V-Shape, Cone, Strawberry''')
                st.markdown("#### Lower body is smaller than the upper body")
                st.markdown("#### Shoulders are wider than hips")
                st.markdown("#### Waist is not prominent")
            if st.button("Show recommended dresses",type="primary"):
                confirmaton()


if __name__ == "__main__":
    app()