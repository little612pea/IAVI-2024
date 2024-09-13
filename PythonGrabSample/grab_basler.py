'''
insatll pypylon and opencv-python
run in cmd :
pip install  -i https://mirrors.zju.edu.cn/pypi/web/simple pypylon opencv-python
''' 
from pypylon import pylon
from pypylon import genicam
import sys
import cv2
import os
from datetime import datetime
#================================================================================
#Grab images from the first camera,presss 's' to save a image, press 'q' to quit
#================================================================================

default_cameraSettings={
    'r_balance':1,
    'g_balance':1,
    'b_balance':1,
    'gain_db':0,
    'exposure_time':500000,
    'PixelFormat':'RGB8',
    'gamma':1.0

}
def OpenFirstCamera():
    devices = pylon.TlFactory.GetInstance().EnumerateDevices()
    if(len(devices)==0):
        return None
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(devices[0]))
    camera.Open()
    return camera
def OpenCameraBySerialnumber(required_serial_number):
    devices = pylon.TlFactory.GetInstance().EnumerateDevices()
    for device in devices:
        if device.GetSerialNumber() == required_serial_number:
            camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device))
            camera.Open()
            return camera
    return None
def SetCamera(camera,cameraSettings):
    #set whitebalance
    camera.PixelFormat.SetValue(cameraSettings['PixelFormat'])
    camera.UserSetSelector = "Default"
    camera.UserSetLoad.Execute()
    camera.BalanceWhiteAuto.SetValue("Off")
    camera.BalanceRatioSelector.SetValue = "Red"
    camera.BalanceRatio.SetValue(cameraSettings['r_balance'])
    camera.BalanceRatioSelector.SetValue = "Green"
    camera.BalanceRatio.SetValue(cameraSettings['g_balance'])
    camera.BalanceRatioSelector.SetValue = "Blue"
    camera.BalanceRatio.SetValue(cameraSettings['b_balance'])
    #exposure time us
    camera.ExposureTime.SetValue(cameraSettings['exposure_time'])
    camera.Gain.Value=cameraSettings['gain_db']
def PrintInfo(camera):
    print("---------------------INFO OF CAMERA-----------------------")
    print("camera Model:",camera.GetDeviceInfo().GetModelName())
    print("Series_Number:",camera.GetDeviceInfo().GetSerialNumber())
    camera.BalanceRatioSelector.SetValue = "Red"
    print("White Balance R:",camera.BalanceRatio.Value)
    camera.BalanceRatioSelector.SetValue ="Green"
    print("White Balance G:",camera.BalanceRatio.Value)
    camera.BalanceRatioSelector.SetValue = "Blue"
    print("White Balance_B:",camera.BalanceRatio.Value)
    print("Gamma Value: ", camera.Gamma.Value)
    print("Exposure Time:",camera.ExposureTime.GetValue(),"us")
    print("Gain:",camera.Gain.Value,"dB")

    print("---------------------FINISH-----------------------")
if __name__=='__main__':
    folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_root=os.path.join(os.getcwd(),f"{folder_name}/")
    exitCode = 0
    try:
        Img = pylon.PylonImage()
        camera=OpenFirstCamera()
        if(camera==None):
            print("WRONG,no camera connected")
        SetCamera(camera,default_cameraSettings)
        PrintInfo(camera)
        '''The parameter MaxNumBuffer can be used to control the count of buffers
        allocated for grabbing. The default value of this parameter is 10.'''
        # camera.MaxNumBuffer = 5

        '''The camera device is parameterized with a default configuration which
        sets up free-running continuous acquisition.'''
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        image_count = 0
        while camera.IsGrabbing():
            # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                # Access the image data.
                img = cv2.cvtColor(grabResult.Array, cv2.COLOR_BAYER_RG2RGB)
                cv2.imshow('Grabbed Image', cv2.resize(img,None,None,fx=0.2,fy=0.2))
                k=cv2.waitKey(1)
                if(k==ord('s') or k==ord('S')):
                    os.makedirs(save_root, exist_ok=True)
                    filename = "{}/image_{:04}.png".format(save_root, image_count)
                    Img.AttachGrabResultBuffer(grabResult)
                    Img.Save(pylon.ImageFileFormat_Png, filename)
                    image_count += 1
                    print(f"{image_count}")
                if(k==ord('q') or k==ord('Q')):
                    print(f"save to {save_root}")
                    break
            else:
                print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
            grabResult.Release()
    except genicam.GenericException as e:
        print("An exception occurred.")
        print(e)
        exitCode = 1
    sys.exit(exitCode)