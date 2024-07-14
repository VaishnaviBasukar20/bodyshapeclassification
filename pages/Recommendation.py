import streamlit as st
from utils.utitlity import load_css
from PIL import Image
import os


def app(page = "recommendation"):
    """
    The function `app` in the provided Python code generates a Streamlit web application for
    recommending dresses and showcasing celebrity wardrobes based on body shape.
    
    :param page: The `page` parameter in the `app` function is used to specify the page to be displayed.
    It is set to "recommendation" by default. This parameter is used to customize the page configuration and load the corresponding CSS file based on the specified page, defaults to recommendation (optional)
    """

    # The code snippet is setting up the initial configuration for a Streamlit web application.
    n_cols = 4
    size = 400
    with open("result.txt","r") as file:
        shape = file.read().strip().lower()
    st.set_page_config(page_icon="assests\logo.png",
                       page_title="Myntra", layout="wide",initial_sidebar_state="collapsed")
    load_css(f"style\{page}.css")

    # It is Python script using Streamlit to create a web application interface for displaying recommended dresses based on body shape.
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
                
        st.html(f'''<h2 style="padding-left: 20px;" ><u>Recommended dresses for</u>: <i>{shape.upper()}</i></h2>''')

        # The code below generates a user interface for displaying images of dresses and celebrity wardrobes based on different body shapes.
        with st.container(border=False):
            st.html("""<div id = 'inner-container'></div>""")
            with st.container(border=True,height=size+50):
                st.html("""<div id = 'img-container-1'></div>""")
                my_imgs = os.listdir(f"assests\{shape}\western\dresses")
                cols = st.columns(n_cols)
                rem = len(my_imgs)%n_cols
                n=0
                for i in range(len(my_imgs)-rem):
                    img = Image.open(f"assests\{shape}\western\dresses\{my_imgs[i]}").resize((size,size))
                    cols[n].image(img)
                    if n%n_cols==n_cols-1:
                        n=0
                    else:
                        n+=1
                if rem!=0:
                    for i in range(rem):
                        img = Image.open(f"assests\{shape}\western\dresses\{my_imgs[len(my_imgs)-rem+i]}").resize((size,size))
                        cols[i].image(img)

            # It displays a list of influencers' wardrobes based on different body shapes.
            st.html("<h4>You can follow following influencer's wardrobe:</h4>")
            if shape=="pear":
                celeb = ["Mrunal Thakur","Parineeti Chopra"]
            elif shape=="rectangle":
                celeb = ["Alia Bhat","Zendaya"]
            elif shape=="apple":
                celeb = ["Rashmi Desai","Kate Winslet"]
            elif shape=="hourglass":
                celeb = ["Tamannah Bhatia","Urvashi Rautela"]
            elif shape=="inverted triangle":
                celeb = ["Sonakshi Sinha"]
            
            # Now generating a user interface for displaying images of celebrities in traditional and western attire.
            for c in range(len(celeb)):
                if c!=0:
                    st.write("")
                st.html(f'''<ul><li><h5 style="color:#f3ecf0"><u>{celeb[c]}</u></h5></li></ul>''')
                with st.expander("Show"):
                    st.divider()

                    # We are checking if a specific directory exists based on the body shape and clothing type. If the directory exists, it creates tabs for
                    # "Traditional" and "Western" clothing options for a celebrity wardrobe.
                    if os.path.exists(f"assests\{shape}\\traditional\celeb {c+1}"):
                        tabs = st.tabs(["Traditional","Western"])
                        with tabs[0]:
                            tabT = "traditional"
                            with st.container(border=True,height=size+50):
                                st.html("""<div id = 'img-container-1'></div>""")
                                path=f"assests\{shape}\{tabT}\celeb {c+1}"
                                my_imgs = os.listdir(path)
                                cols = st.columns(n_cols)
                                rem = len(my_imgs)%n_cols
                                n=0
                                for i in range(len(my_imgs)-rem):
                                    img = Image.open(f"{path}\{my_imgs[i]}").resize((size,size))
                                    cols[n].image(img)
                                    if n%n_cols==n_cols-1:
                                        n=0
                                    else:
                                        n+=1
                                if rem!=0:
                                    for i in range(rem):
                                        img = Image.open(f"{path}\{my_imgs[len(my_imgs)-rem+i]}").resize((size,size))
                                        cols[i].image(img)
                        if os.path.exists(f"assests\{shape}\western\celeb {c+1}"):
                            with tabs[1]:
                                tabT = "western"
                                with st.container(border=True,height=size+50):
                                    st.html("""<div id = 'img-container-1'></div>""")
                                    path=f"assests\{shape}\{tabT}\celeb {c+1}"
                                    my_imgs = os.listdir(path)
                                    cols = st.columns(n_cols)
                                    rem = len(my_imgs)%n_cols
                                    n=0
                                    for i in range(len(my_imgs)-rem):
                                        img = Image.open(f"{path}\{my_imgs[i]}").resize((size,size))
                                        cols[n].image(img)
                                        if n%n_cols==n_cols-1:
                                            n=0
                                        else:
                                            n+=1
                                    if rem!=0:
                                        for i in range(rem):
                                            img = Image.open(f"{path}\{my_imgs[len(my_imgs)-rem+i]}").resize((size,size))
                                            cols[i].image(img)
                    else:

                        # If not , it only creates a tab for "Western" clothing options for a celebrity wardrobe.
                        if os.path.exists(f"assests\{shape}\western\celeb {c+1}"):
                            tabs = st.tabs(["Western"])
                            with tabs[0]:
                                tabT = "western"
                                with st.container(border=True,height=size+50):
                                    st.html("""<div id = 'img-container-1'></div>""")
                                    path=f"assests\{shape}\{tabT}\celeb {c+1}"
                                    my_imgs = os.listdir(path)
                                    cols = st.columns(n_cols)
                                    rem = len(my_imgs)%n_cols
                                    n=0
                                    for i in range(len(my_imgs)-rem):
                                        img = Image.open(f"{path}\{my_imgs[i]}").resize((size,size))
                                        cols[n].image(img)
                                        if n%n_cols==n_cols-1:
                                            n=0
                                        else:
                                            n+=1
                                    if rem!=0:
                                        for i in range(rem):
                                            img = Image.open(f"{path}\{my_imgs[len(my_imgs)-rem+i]}").resize((size,size))
                                            cols[i].image(img)
            
    st.header("#")

if __name__ == "__main__":
    app()