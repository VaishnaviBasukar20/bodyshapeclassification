import streamlit as st



def load_css(file_path) -> None:
    """
    The function `load_css` reads a CSS file and injects its contents into an HTML file.
    :param file_path: The `file_path` parameter in the `load_css` function is a string that represents
    the path to the CSS file that you want to load and inject into an HTML document. This function reads
    the content of the CSS file and embeds it within a `<style>` tag in the HTML document
    """
    with open(file_path) as f:
        st.html(f'<style>{f.read()}</style>')

def cmToinche(cm:float) -> float:
    """
    The function `cmToinche` converts centimeters to inches by dividing by 2.54 and rounding to the
    nearest whole number.
    
    :param cm: The parameter `cm` in the `cmToinche` function represents the length in centimeters that
    you want to convert to inches
    :type cm: float
    :return: An integer value representing the conversion of centimeters to inches is being returned.
    """
    return round(cm/2.54,2)

def shapeIdentifiyer(bust:float, waist:float , hip:float, by:str = "inches") -> str:
    """
    The function `shapeIdentifiyer` determines a person's body shape based on their bust, waist, and hip
    measurements.
    
    :param bust:
    :type bust: int
    :param waist:
    :type waist: int
    :param hip:
    :type hip: int
    :return: a string that represents the shape of a person based on their bust, waist, and hip measurements. The possible return values are "Rectangle", "Hourglass", "Apple", "Inverted triangle", or "Pear" depending on the conditions specified in the function.
    """
    if by.lower() == "inches":
        if abs(bust-hip)<4:
            if 4>min(hip,bust)-waist>-2:
                return "Rectangle"
            else:
                if min(hip,bust)-waist>3:
                    return "Hourglass"
                else:
                    return "Apple"
        else:
            if bust>hip:
                if bust-waist < -1:
                    return "Apple"
                else:
                    return "Inverted triangle"
            else:
                if hip-waist < -1:
                    return "Apple"
                else:
                    return "Pear"
    elif by.lower() == "cm":
        if abs(bust-hip)<10.16:
            if 10.16>min(hip,bust)-waist>-5.08:
                return "Rectangle"
            else:
                if min(hip,bust)-waist>-7.62:
                    return "Hourglass"
                else:
                    return "Apple"
        else:
            if bust>hip:
                if bust-waist < -2.54:
                    return "Apple"
                else:
                    return "Inverted triangle"
            else:
                if hip-waist < -2.54:
                    return "Apple"
                else:
                    return "Pear"
                
def predictShape(img_path):
    from keras.models import load_model
    import cv2
    import numpy as np

    def apply_canny(image):
        image = cv2.resize(image, (224, 224))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.4)
        edges = cv2.Canny(blurred_image, 80, 150)
        edges_3_channel = cv2.merge([edges, edges, edges])
        return edges_3_channel / 255.0

    # Load the model
    model = load_model(r'ml\model\resnet50_body_shape_classification_model.keras')

    # Load an image file that you want to test
    img = cv2.imread(img_path)

    # Apply the Canny edge detection function
    processed_img = apply_canny(img)

    # Expand dimensions to match the input shape expected by the model
    img_array = np.expand_dims(processed_img, axis=0)

    # Make predictions
    predictions = model.predict(img_array)

    # Decode the predictions
    predicted_class = np.argmax(predictions, axis=1)

    # Assuming you have a list of class labels
    class_labels = ["apple", "pear", "rectangle", "inverted triangle", "hourglass"]

    # Get the human-readable class label
    predicted_label = class_labels[predicted_class[0]]
    
    return predicted_label


# if __name__=="__main__":
#     print(predictShape(r"assests\test.jpg"))