import streamlit as st
import streamlit_folium as sf
import numpy as np
import pandas as pd
from img2loc_GPT4V import GPT4v2Loc
import folium
from folium.plugins import HeatMap
from folium.features import DivIcon
from geopy.distance import geodesic
import base64
    
# In the sidebar, add the widgets for the app
# load the logo image and convert it to base64
logo = open("./static/logo_clipped.png", 'rb').read()
img_base64 = base64.b64encode(logo).decode('utf-8')
st.set_page_config(page_title="Img2Loc", page_icon=":earth_americas:")

st.sidebar.markdown(
    f"<img src='data:image/png;base64,{img_base64}' style='display: block; margin-left: auto; margin-right: auto; width: 150px;'>", 
    unsafe_allow_html=True
)


uploaded_file = st.sidebar.file_uploader("Upload an image")
openai_api_key = st.sidebar.text_input("OpenAI API Key", "xxxxxxxxx", key="chatbot_api_key", type="password")
nearest_neighbor = st.sidebar.radio("Use nearest neighbor search?", ("Yes", "No"))
num_nearest_neighbors = None  # Default value

if nearest_neighbor == "Yes":
    num_nearest_neighbors = st.sidebar.number_input("Number of nearest neighbors", value=16)
    
farthest_neighbor = st.sidebar.radio("Use farthest neighbor search?", ("Yes", "No"))

num_farthest_neighbors = None  # Default value

if farthest_neighbor == "Yes":
    num_farthest_neighbors = st.sidebar.number_input("Number of farthest neighbors", value=16)

# Create two columns in the sidebar
col1, col2 = st.sidebar.columns(2)

# Add input for real latitude and longitude in the columns
real_latitude = col1.number_input("Enter the latitude")
real_longitude = col2.number_input("Enter the longitude")

submit = st.sidebar.button("Submit")


# Add the title and maps
st.markdown("<h1 style='text-align: center;'>Img2Loc</h1>", unsafe_allow_html=True)
st.markdown("---")  # Dash line separation


if submit:
    my_bar = st.progress(0, text="Starting Analysis...")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    if uploaded_file is None:
        st.info("Please upload an image.")
        st.stop()
    my_bar.progress(10, text="Loading Required Resources...")
    GPT_Agent = GPT4v2Loc(device="cpu")

    # Get the name of the uploaded file
    img_name = uploaded_file.name
    
    if real_latitude != 0.0 and real_longitude != 0.0:
        true_lat = real_latitude
        true_long = real_longitude
    
    num_neighbors = 16
    use_database_search = True if nearest_neighbor == "Yes" or farthest_neighbor == "Yes" else False
    if use_database_search == "Yes":
        my_bar.progress(25, text="Finding nearest and farthest neighbors...")
    
    my_bar.progress(50, text="Transforming Image...")
    GPT_Agent.set_image_app(uploaded_file, use_database_search = use_database_search, num_neighbors = num_nearest_neighbors, num_farthest = num_farthest_neighbors)
    my_bar.progress(75, text="Obtaining Locations...")
    coordinates = GPT_Agent.get_location(openai_api_key, use_database_search=True)
    my_bar.progress(90, text="Generating Map...")
    
    lat_str, lon_str = coordinates.split(',')
    latitude = float(lat_str)
    longitude = float(lon_str)

    # Display the maps with captions
    col1, mid, col2 = st.columns([1,0.1,1])  # Create three columns
    
    
    # Map for the nearest neighbor points
    if nearest_neighbor == "Yes":
        m1 = folium.Map(width=320,height=200, location=GPT_Agent.neighbor_locations_array[0], zoom_start=4)
        folium.TileLayer('cartodbpositron').add_to(m1)
        for i in GPT_Agent.neighbor_locations_array:
            print(i)
            folium.Marker(i, tooltip='({}, {})'.format(i[0], i[1]), icon=folium.Icon(color="green", icon="compass", prefix="fa")).add_to(m1)
        with col1:
            st.markdown("<h3 style='text-align: center;'>Nearest Neighbor Points Map</h3>", unsafe_allow_html=True)
            sf.folium_static(m1, height=200)

    # Map for the farthest neighbor points
    if farthest_neighbor == "Yes":
        m2 = folium.Map(width=320,height=200, location=GPT_Agent.farthest_locations_array[0], zoom_start=3)
        folium.TileLayer('cartodbpositron').add_to(m2)
        for i in GPT_Agent.farthest_locations_array:
            folium.Marker(i, tooltip='({}, {})'.format(i[0], i[1]), icon=folium.Icon(color="blue", icon="compass", prefix="fa")).add_to(m2)
        with col2:
            st.markdown("<h3 style='text-align: center;'>Farthest Neighbor Points Map</h3>", unsafe_allow_html=True)
            sf.folium_static(m2, height=200)

    # Map for the predicted point, the true point, and the distance between them
    m3 = folium.Map(width=1000,height=400, location=[latitude, longitude], zoom_start=12)

    folium.Marker([latitude, longitude], tooltip='Img2Loc Location', popup=f'latitude: {latitude}, longitude: {longitude}', icon=folium.Icon(color="red", icon="map-pin", prefix="fa")).add_to(m3)
    # line = folium.PolyLine(locations=[[latitude, longitude], [true_lat, true_long]], color='black', dash_array='5', weight=2).add_to(m3)
    folium.TileLayer('cartodbpositron').add_to(m3)
    
    st.markdown("<h3 style='text-align: center;'>Prediction Map</h3>", unsafe_allow_html=True)
    sf.folium_static(m3)
    my_bar.progress(100, text="Done!")
        
else:
    # load the background image and convert it to base64
    bg_image = open("./static/figure3.jpg", 'rb').read()
    bg_base64 = base64.b64encode(bg_image).decode('utf-8')
    # while the user has not submitted the form, display the background image
    st.markdown(
        f"""
        <style>
        .stApp::before {{
            content: "";
            position: absolute;
            top: -10%;
            left: 35%;  /* Adjust this to center the image */
            height: 100%;
            width: 50%;
            background: url("data:image/png;base64,{bg_base64}");
            background-repeat: no-repeat;
            background-position: center;
            background-size: contain;
        }}
        .stApp {{
            background-blend-mode: multiply;
            background-color: rgba(255, 255, 255, 0.8);  /* white with 50% transparency */
        }}
        </style>
        """, 
        unsafe_allow_html=True  
    ) 