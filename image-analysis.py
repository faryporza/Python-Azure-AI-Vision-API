from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
import sys
from matplotlib import pyplot as plt
from azure.core.exceptions import HttpResponseError
import requests

# นำเข้า namespaces
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

def main():
    global cv_client

    # ล้างหน้าจอคอนโซล
    os.system('cls' if os.name=='nt' else 'clear')

    try:
        # รับการตั้งค่าการกำหนดค่า
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        # รับรูปภาพ
        image_file = 'images/street.jpg'
        if len(sys.argv) > 1:
            image_file = sys.argv[1]

        with open(image_file, "rb") as f:
            image_data = f.read()

        # ตรวจสอบสิทธิ์ Azure AI Vision client
        cv_client = ImageAnalysisClient(
            endpoint=ai_endpoint,
            credential=AzureKeyCredential(ai_key)
        )
        
        # วิเคราะห์รูปภาพ
        AnalyzeImage(image_file, image_data, cv_client)
        
    except Exception as ex:
        print(ex)


def AnalyzeImage(image_filename, image_data, cv_client):
    print('\nกำลังวิเคราะห์รูปภาพ...')

    try:
        # รับผลลัพธ์พร้อมฟีเจอร์ที่ระบุให้ดึงข้อมูล
        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.DENSE_CAPTIONS,
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,
                VisualFeatures.PEOPLE
            ]
        )

        # แสดงผลลัพธ์การวิเคราะห์
        # รับคำบรรยายรูปภาพ
        if result.caption is not None:
            print(f"\nคำบรรยาย: '{result.caption.text}' (ความเชื่อมั่น: {result.caption.confidence:.2f})")

        # รับคำบรรยายแบบหนาแน่น
        if result.dense_captions is not None:
            print("\nคำบรรยายแบบหนาแน่น:")
            for caption in result.dense_captions.list:
                print(f"  '{caption.text}' (ความเชื่อมั่น: {caption.confidence:.2f})")

        # รับแท็กรูปภาพ
        if result.tags is not None:
            print("\nแท็ก:")
            for tag in result.tags.list:
                print(f"  '{tag.name}' (ความเชื่อมั่น: {tag.confidence:.2f})")

        # รับวัตถุในรูปภาพ
        if result.objects is not None:
            print("\nวัตถุ:")
            # เตรียมรูปภาพสำหรับการวาด
            image = Image.open(image_filename)
            fig = plt.figure(figsize=(image.width/100, image.height/100))
            plt.axis('off')
            draw = ImageDraw.Draw(image)
            color = 'cyan'

            for detected_object in result.objects.list:
                print(f"  '{detected_object.tags[0].name}' (ความเชื่อมั่น: {detected_object.tags[0].confidence:.2f})")
                
                # วาดกรอบขอบเขตวัตถุ
                r = detected_object.bounding_box
                bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
                draw.rectangle(bounding_box, outline=color, width=3)
                plt.annotate(detected_object.tags[0].name, (r.x, r.y), backgroundcolor=color)

            # บันทึกรูปภาพที่มีคำอธิบายประกอบ
            plt.imshow(image)
            plt.tight_layout(pad=0)
            outputfile = 'objects.jpg'
            fig.savefig(outputfile, bbox_inches='tight', dpi=300)
            print(f'  ผลลัพธ์ถูกบันทึกใน {outputfile}')

        # รับบุคคลในรูปภาพ
        if result.people is not None:
            print("\nบุคคล:")
            # เตรียมรูปภาพสำหรับการวาด
            image = Image.open(image_filename)
            fig = plt.figure(figsize=(image.width/100, image.height/100))
            plt.axis('off')
            draw = ImageDraw.Draw(image)
            color = 'cyan'

            for detected_person in result.people.list:
                print(f"  บุคคล (ความเชื่อมั่น: {detected_person.confidence:.2f})")
                
                # วาดกรอบขอบเขตบุคคล
                r = detected_person.bounding_box
                bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
                draw.rectangle(bounding_box, outline=color, width=3)
                # ไม่แสดงข้อความที่กล่องพื้นที่
                # plt.annotate('บุคคล', (r.x, r.y), backgroundcolor=color)

            # บันทึกรูปภาพที่มีคำอธิบายประกอบ
            plt.imshow(image)
            plt.tight_layout(pad=0)
            outputfile = 'people.jpg'
            fig.savefig(outputfile, bbox_inches='tight', dpi=300)
            print(f'  ผลลัพธ์ถูกบันทึกใน {outputfile}')

        print("\nการวิเคราะห์เสร็จสิ้น")

    except HttpResponseError as e:
        print(f"Status code: {e.status_code}")
        print(f"Reason: {e.reason}")
        print(f"Message: {e.error.message}")
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    main()
