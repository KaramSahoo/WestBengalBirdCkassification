# import streamlit as st
# from PIL import Image
# from torchvision import models, transforms
# import torch
# import pandas
# import pickle
# from streamlit_option_menu import option_menu

# with open('classlabels.pkl', 'rb') as f:
#     class_names = pickle.load(f)


# def predict(image_path):
#     model = torch.load('Googlenet_50_epochs',
#                        map_location=torch.device('cpu'))

#     # https://pytorch.org/docs/stable/torchvision/models.html
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )])

#     img = Image.open(image_path)
#     batch_t = torch.unsqueeze(transform(img), 0)

#     model.eval()
#     outputs = model(batch_t)
#     _, predicted = torch.max(outputs, 1)
#     title = [class_names[x] for x in predicted]
#     prob = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
#     classes = pandas.read_csv('bird_dataset.csv', header=None)
#     # print("prob: ", float(max(prob)))
#     # print("title: ", title[0])
#     # print("name: ", classes[0][int(title[0])-1].split('.')[0])
#     return (float(max(prob)), classes[0][int(title[0])-1].split('.')[0])


# st.set_option('deprecation.showfileUploaderEncoding', False)

# selected = option_menu(
#     menu_title=None,
#     options=["Home", "Project"],
#     orientation="horizontal",
#     default_index=0,
#     menu_icon="cast",
#     icons=['house-fill', 'gear-fill'],
#     styles={
#         "nav-link-selected": {"background-color": "#1c0d3e"},
#     }
# )

# if selected == "Project":
#     # st.title("Karam and Ishan's Simple Image Classification App")
#     st.write("")

#     file_up = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

#     if file_up is not None:
#         image = Image.open(file_up)

#         col1, col2 = st.columns([0.5, 0.5])
#         with col1:
#             st.image(image, caption='Uploaded Image.', use_column_width=True)
#             st.write("")

#         with col2:
#             st.write("Your results are served here...")
#             score, bird_name = predict(file_up)
#             # st.write(results)
#             if score > 60:
#                 st.write("Prediction (name): ",
#                          score, ",   \nScore: ", bird_name)
#             else:
#                 st.write("No such bird in database!")

# elif selected == "Home":
#     st.title("West bengal bird species classification project")


import streamlit as st
from PIL import Image
from torchvision import models, transforms
import torch
import pandas
import pickle
# from streamlit_option_menu import option_menu

with open('classlabels.pkl', 'rb') as f:
    class_names = pickle.load(f)


def predict(image_path):
    model = torch.load('Googlenet_50_epochs',
                       map_location=torch.device('cpu'))

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
    outputs = model(batch_t)
    _, predicted = torch.max(outputs, 1)
    title = [class_names[x] for x in predicted]
    prob = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    classes = pandas.read_csv('bird_dataset.csv', header=None)
    # print("prob: ", float(max(prob)))
    # print("title: ", title[0])
    # print("name: ", classes[0][int(title[0])-1].split('.')[0])
    return (float(max(prob)), classes[0][int(title[0])-1].split('.')[0])


st.set_option('deprecation.showfileUploaderEncoding', False)

# selected = option_menu(
#     menu_title=None,
#     options=["Home", "Project"],
#     orientation="horizontal",
#     default_index=0,
#     menu_icon="cast",
#     icons=['house-fill', 'gear-fill'],
#     styles={
#         "nav-link-selected": {"background-color": "#1c0d3e"},
#     }
# )

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
        score, bird_name = predict(file_up)
        # st.write(results)
        if score > 60:
            st.write("Prediction (name): ",
                     score, ",   \nScore: ", bird_name)
        else:
            st.write("No such bird in database!")
