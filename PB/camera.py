import cv2
import sys
import numpy as np
import torch 
from torchvision.models.segmentation import fcn_resnet50

## Artbooth basic camera settings
color = (255,255,255)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_size = 1
y_offset = 100



## art functions

#option 1
def pencil_sketch(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale = 256)
    return cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)

#option 2 
def watercolor(frame):
    return cv2.stylization(frame, sigma_s=60, sigma_r=0.6)

#option 3 van ghoh
def load_style_transfer(style= 'starry_night'):
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    model = torch.hub.load(
        'pytorch/vision',
        'fast_neural_style',
        style_name=style,
        pretrained = True
    )
    return model.eval()

vg_model = load_style_transfer('starry_night')

#option 4 monet
monet_model=load_style_transfer('masaic')

#option 5 candy
# candy_model = load_style_transfer('candy')

#option 6 rainprincess
# rainp_model = load_style_transfer('rain_princess')


def apply_nst(frame, model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = torch.from_numpy(frame).permute(2,0,1).float().div(255).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        
    output = output.squeeze().permute(1,2,0).numpy() * 255
    return cv2.cvtColor(output.astype('uint8'), cv2.COLOR_RGB2BGR)
    return output


cap =cv2.VideoCapture(0)
current_effect = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("error. now exiting.")
        break
    
    frame = cv2.resize(frame, (650, 400))
    frame = cv2.flip(frame, 1)
    
    cv2.imshow('''Welcome to my Artbooth! Press 'S' to capture. Press 'Q' to quit the program!
               Styles: 1= Normal, 2= Pencil, 3= Watercolor, 4= VanGogh, 5=Monet''', frame)
    
    if current_effect == 'sketch':
        frame = pencil_sketch(frame)
    elif current_effect == 'watercolor':
        frame = watercolor(frame)
    elif current_effect == 'vangogh':
        frame =apply_nst(frame, vg_model)
    elif current_effect == 'monet':
        frame =apply_nst(frame, monet_model)
        
    
    key = cv2.waitKey(1)
    if key == ord('1'):
        current_effect = None
    elif key == ord('2'):
        current_effect = 'sketch'
    elif key == ord('3'):
        current_effect = 'watercolor'
    elif key == ord('4'):
        current_effect = 'vangogh'
    elif key == ord('5'):
        current_effect = 'monet'
        
    elif key == ord('s'):
        cv2.imwrite("picture.png", frame)
        print("captured picture saved as 'picture.png'.")
    elif key == ord('q'):
        exit_booth = frame.copy()
        cv2.putText(exit_booth, "Thank you for testing Artbooth! Hope to see you again!", (50,y_offset), font, font_scale, color, font_size)
        cv2.imshow("Goodbye!", exit_booth)
        cv2.waitKey(3000)
        break
    
cap.release()
cv2.destroyAllWindows()







 

