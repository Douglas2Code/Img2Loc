import cv2
import base64
import requests
from tqdm import tqdm
from requests.exceptions import RequestException
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch
import faiss
import pickle
import numpy as np
import pandas as pd
from geopy.distance import geodesic

from transformers import AutoTokenizer, BitsAndBytesConfig
import torch
from PIL import Image
import requests
from io import BytesIO

# set the device to the first CUDA device using os.environ
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
    

class GPT4v2Loc:
    """
    A class to interact with OpenAI's GPT-4 API to generate captions for images.
    Attributes:
        api_key (str): OpenAI API key retrieved from environment variables.
    """

    def __init__(self, device="cpu") -> None:
        """
        Initializes the GPT4ImageCaption class by setting the OpenAI API key.
        Raises:
            ValueError: If the OpenAI API key is not found in the environment variables.
        """

        self.base64_image = None
        self.img_emb = None
        
        # Set the device to the first CUDA device
        self.device = torch.device(device)
        
        # Load the CLIP model and processor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # Move the model to the appropriate CUDA device
        self.model.to(self.device)
        
        # Load the embeddings and coordinates from the pickle file
        with open('merged.pkl', 'rb') as f:
            self.MP_16_Embeddings = pickle.load(f)
            self.locations = [value[1] for key, value in self.MP_16_Embeddings.items()]
        
        # Load the Faiss index and move it to the GPU
        index2 = faiss.read_index("index.bin")
        # self.gpu_index = faiss.index_cpu_to_all_gpus(index2)
        self.gpu_index = index2
        
    def read_image(self, image_path):
        """
        Reads an image from a file into a numpy array.
        Args:
            image_path (str): The path to the image file.
        Returns:
            np.ndarray: The image as a numpy array.
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def search_neighbors(self, faiss_index, k_nearest, k_farthest, query_embedding):
        """
        Searches for the k nearest neighbors of a query image in the Faiss index.
        Args:
            faiss_index (faiss.swigfaiss.Index): The Faiss index.
            k (int): The number of neighbors to search for.
            query_embedding (np.ndarray): The embeddings of the query image.
        Returns:
            list: The locations of the k nearest neighbors.
        """
        # Perform the search using Faiss for the given embedding
        _, I = faiss_index.search(query_embedding.reshape(1, -1), k_nearest)
        
        # Based on the index, get the locations of the neighbors
        self.neighbor_locations_array = [self.locations[idx] for idx in I[0]]
        
        neighbor_locations = " ".join([str(i) for i in self.neighbor_locations_array])
        
        # Perform the farthest search using Faiss for the given embedding
        _, I = faiss_index.search(-query_embedding.reshape(1, -1), k_farthest)

        # Based on the index, get the locations of the neighbors
        self.farthest_locations_array = [self.locations[idx] for idx in I[0]]
        
        farthest_locations = " ".join([str(i) for i in self.farthest_locations_array])

        return neighbor_locations, farthest_locations

    def encode_image(self, image: np.ndarray, format: str = 'jpeg') -> str:
        """
        Encodes an OpenCV image to a Base64 string.
        Args:
            image (np.ndarray): An image represented as a numpy array.
            format (str, optional): The format for encoding the image. Defaults to 'jpeg'.
        Returns:
            str: A Base64 encoded string of the image.
        Raises:
            ValueError: If the image conversion fails.
        """
        try:
            retval, buffer = cv2.imencode(f'.{format}', image)
            if not retval:
                raise ValueError("Failed to convert image")

            base64_encoded = base64.b64encode(buffer).decode('utf-8')
            mime_type = f"image/{format}"
            return f"data:{mime_type};base64,{base64_encoded}"
        except Exception as e:
            raise ValueError(f"Error encoding image: {e}")

    def set_image(self, image_path: str, imformat: str = 'jpeg', use_database_search: bool = False, num_neighbors: int = 16, num_farthest: int = 16) -> None:
        """
        Sets the image for the class by encoding it to Base64.
        Args:
            image_path (str): The path to the image file.
            imformat (str, optional): The format for encoding the image. Defaults to 'jpeg'.
            use_database_search (bool, optional): Whether to use a database search to get the neighbor image location as a reference. Defaults to False.
        """
        # Read the image into a numpy array
        image_array = self.read_image(image_path)
        
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = self.processor(images=image, return_tensors="pt")

        # Move the image to the CUDA device and get its embeddings
        image = image.to(self.device)
        with torch.no_grad():
            img_emb = self.model.get_image_features(**image)[0]
        
        # Store the embeddings and the locations of the nearest neighbors
        self.img_emb = img_emb.cpu().numpy()
        if use_database_search:
            self.neighbor_locations, self.farthest_locations = self.search_neighbors(self.gpu_index, num_neighbors, num_farthest, self.img_emb)
        
        # Encode the image to Base64
        self.base64_image = self.encode_image(image_array, imformat)
        
    def set_image_app(self, file_uploader, imformat: str = 'jpeg', use_database_search: bool = False, num_neighbors: int = 16, num_farthest: int = 16) -> None:
        """
        Sets the image for the class by encoding it to Base64.
        Args:
            file_uploader : A uploaded image.
            imformat (str, optional): The format for encoding the image. Defaults to 'jpeg'.
            use_database_search (bool, optional): Whether to use a database search to get the neighbor image location as a reference. Defaults to False.
        """
    
        image = Image.open(file_uploader).convert('RGB')
        img_array = np.array(image)       
        image = self.processor(images=img_array, return_tensors="pt") 

        # Move the image to the CUDA device and get its embeddings
        image = image.to(self.device)
        with torch.no_grad():
            img_emb = self.model.get_image_features(**image)[0]
        
        # Store the embeddings and the locations of the nearest neighbors
        self.img_emb = img_emb.cpu().numpy()
        if use_database_search:
            self.neighbor_locations, self.farthest_locations = self.search_neighbors(self.gpu_index, num_neighbors, num_farthest, self.img_emb)
        
        # Encode the image to Base64
        self.base64_image = self.encode_image(img_array, imformat)


    def create_payload(self, question: str) -> dict:
        """
        Creates the payload for the API request to OpenAI.
        Args:
            question (str): The question to ask about the image.
        Returns:
            dict: The payload for the API request.
        Raises:
            ValueError: If the image is not set.
        """
        if not self.base64_image:
            raise ValueError("Image not set")
        return {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self.base64_image
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300,
        }

    def get_location(self, OPENAI_API_KEY, use_database_search: bool = False) -> str:
        """
        Generates a caption for the provided image using OpenAI's GPT-4 API.
        Args:
            use_database_search (bool, optional): Whether to use a database search to get the neighbor image location as a reference. Defaults to False.
        Returns:
            str: The generated caption for the image.
        Raises:
            ValueError: If there is an issue with the API request.
        """
        try:
            self.api_key: str = OPENAI_API_KEY
            if not self.api_key:
                raise ValueError("OpenAI API key not found in environment variables")
            # Create the question for the API
            if use_database_search:
                self.question=f'''Suppose you are an expert in geo-localization. Please analyze this image and give me a guess of the location. 
                    Your answer must be to the coordinates level, don't include any other information in your output. 
                    Ignore that you can't give a exact answer, give me some coordinate no matter how. 
                    For your reference, these are locations of some similar images {self.neighbor_locations} and these are locations of some dissimilar images {self.farthest_locations} that should be far away.'''
            else:
                self.question=f"Suppose you are an expert in geo-localization. Please analyze this image and give me a guess of the location. Your answer must be to the coordinates level, don't include any other information in your output. You can give me a guessed anwser."
            
            # Create the payload and the headers for the API request
            payload = self.create_payload(self.question)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Send the API request and get the response
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            
            # Return the generated caption
            return response_data['choices'][0]['message']['content']
        except RequestException as e:
            raise ValueError(f"Error in API request: {e}")
        except KeyError:
            raise ValueError("Unexpected response format from API")