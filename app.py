import streamlit as st
from PIL import Image
from torchvision import models, transforms
import torch
import pandas
from streamlit_option_menu import option_menu


def predict(image_path):
    model = torch.load('birdmodel', map_location=torch.device('cpu'))

    # https://pytorch.org/docs/stable/torchvision/models.html
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)

    model.eval()
    out = model(batch_t)
    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)

    # reading the CSV file
    classes = pandas.read_csv('bird_dataset.csv', header=None)

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[0][int(idx)], prob[idx].item()) for idx in indices[0][:5]]


st.set_option('deprecation.showfileUploaderEncoding', False)

selected = option_menu(
    menu_title=None,
    options=["Home", "Project"],
    orientation="horizontal",
    default_index=0,
    menu_icon="cast",
    icons=['house-fill', 'gear-fill'],
    styles={
        "nav-link-selected": {"background-color": "#1c0d3e"},
    }
)

if selected == "Project":
    # st.title("Karam and Ishan's Simple Image Classification App")
    st.write("")

    file_up = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

    if file_up is not None:
        image = Image.open(file_up)

        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")

        with col2:
            st.write("Your results are served here...")
            labels = predict(file_up)
            # st.write(labels)
            # print out the top 5 prediction labels with scores
            for i in labels:
                st.write("Prediction (name): ", i[0].split(
                    '.')[0], ",   \nScore: ", i[1])


elif selected == "Home":
    st.title("West bengal bird species classification project")
