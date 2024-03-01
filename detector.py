from pathlib import Path
import face_recognition
import pickle
from collections import Counter
from PIL import Image, ImageDraw
import argparse

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"


## For example few celebrities folder and pictures are used for training, but you might want to/ will use it for personal use, train the model with 5-10 images of youself and your friends, ENJOY!!!! 

def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    names = []
    encodings = []

    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)
        
# encode_known_faces() 
# MY POOR GPU CANNOT HANDLE IT Running multiple times

def recognize_faces( image_location : str, model:str  = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH,):
    with encodings_location.open(mode = 'rb') as f:
        loaded_encodings = pickle.load(f)
        
    input_image = face_recognition.load_image_file(image_location)
    
    input_face_locations = face_recognition.face_locations(input_image, model = model)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)
    
    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)
    
    names = ""
    names_set = set()
    for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        if name not in names_set:  # Check if name is not already in the folder, because DAS not good
            names_set.add(name)  
            if len(names) + len(name) + 1 <= 255:
                names += "_" + name  
        _display_face(draw, bounding_box, name)
        # print(name, bounding_box)

    del draw
    pillow_image.show()
    if not names:
        return "unknown"
    else:
        return names

def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    
    draw.rectangle(((left,top), (right,bottom)),outline = BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left,text_top), (text_right,text_bottom)),
        fill = "blue",
        outline = "blue"
    )
    draw.text(
        (text_left, text_top),
        name,
        fille = "white"
    )

def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    
    votes = Counter(
        name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match
    )
    
    if votes:
        return votes.most_common(1)[0][0]


## REMINDER TO READ COMMENTS IN TRAINING PHASE : 🥸

def name_the_images():
    folder = Path(".")                          #Provide the right path where your images are located
    jpg_files = list(folder.glob("*.jpg"))
    
    new_names = []
    suff = 1
    for img in jpg_files:
        print("here",suff)
        name = recognize_faces(img)
        suff = suff+1 #Please lmk if there is any better way to make it unique, i feel stupid
        print(name)
        if name not in new_names:
            img.rename(name+".png")
        else:
            name = name + suff
            img.rename(name+".png")
        new_names.append(name)

name_the_images()
# DONE DONE DONE


def validate(model: str = "hog"):
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognize_faces(
                image_location=str(filepath.absolute()), model=model
            )


### ----------------- CHECK THIS ------------------- ###

# In case you want to use face recognition for different purpose feel free to use the below code for running and providing command - line args. Thank YOU HEHE!

# if __name__ == "__main__":
#     if args.train:
#         encode_known_faces(model=args.m)
#     if args.validate:
#         cross_validation(model=args.m)
#     if args.test:
#         recognize_faces(image_location=args.f, model=args.m)



# parser = argparse.ArgumentParser(description="Recognize faces in an image")
# parser.add_argument("--train", action="store_true", help="Train on input data")
# parser.add_argument(
#     "--validate", action="store_true", help="Validate trained model"
# )
# parser.add_argument(
#     "--test", action="store_true", help="Test the model with an unknown image"
# )
# parser.add_argument(
#     "-m",
#     action="store",
#     default="hog",
#     choices=["hog", "cnn"],
#     help="Which model to use for training: hog (CPU), cnn (GPU)",
# )
# parser.add_argument(
#     "-f", action="store", help="Path to an image with an unknown face"
# )
# args = parser.parse_args()