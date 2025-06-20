## การวิเคราะห์รูปภาพโดยใช้ Azure AI Vision API

โปรแกรมนี้สามารถวิเคราะห์รูปภาพโดยใช้ Azure AI Vision API เพื่อสร้าง:

- คำบรรยายรูปภาพ พร้อมค่าความเชื่อมั่น
- การระบุแท็กที่เกี่ยวข้องในภาพ
- การตรวจจับวัตถุ (บันทึกผลลัพธ์ใน objects.jpg)
- การตรวจจับบุคคล (บันทึกผลลัพธ์ใน people.jpg)

ตัวอย่าง ผลลัพ

กำลังวิเคราะห์รูปภาพ...

คำบรรยาย: 'a man walking a dog on a leash on a street' (ความเชื่อมั่น: 0.82)

คำบรรยายแบบหนาแน่น:
  'a man walking a dog on a leash on a street' (ความเชื่อมั่น: 0.82)
  'a man walking on a street' (ความเชื่อมั่น: 0.69)
  'a yellow car on the street' (ความเชื่อมั่น: 0.78)
  'a black dog walking on the street' (ความเชื่อมั่น: 0.75)
  'a blurry image of a blue car' (ความเชื่อมั่น: 0.82)
  'a yellow taxi cab on the street' (ความเชื่อมั่น: 0.72)

แท็ก:
  'outdoor' (ความเชื่อมั่น: 1.00)
  'land vehicle' (ความเชื่อมั่น: 0.99)
  'vehicle' (ความเชื่อมั่น: 0.99)
  'building' (ความเชื่อมั่น: 0.99)
  'road' (ความเชื่อมั่น: 0.96)
  'wheel' (ความเชื่อมั่น: 0.95)
  'street' (ความเชื่อมั่น: 0.95)
  'person' (ความเชื่อมั่น: 0.93)
  'clothing' (ความเชื่อมั่น: 0.91)
  'taxi' (ความเชื่อมั่น: 0.91)
  'car' (ความเชื่อมั่น: 0.84)
  'dog' (ความเชื่อมั่น: 0.83)
  'yellow' (ความเชื่อมั่น: 0.77)
  'walking' (ความเชื่อมั่น: 0.74)
  'city' (ความเชื่อมั่น: 0.65)
  'woman' (ความเชื่อมั่น: 0.58)

วัตถุ:
  'car' (ความเชื่อมั่น: 0.72)
  'taxi' (ความเชื่อมั่น: 0.77)
  'person' (ความเชื่อมั่น: 0.78)
  'dog' (ความเชื่อมั่น: 0.54)
  ผลลัพธ์ถูกบันทึกใน objects.jpg

บุคคล:
  บุคคล (ความเชื่อมั่น: 0.95)
  บุคคล (ความเชื่อมั่น: 0.25)
  บุคคล (ความเชื่อมั่น: 0.22)
  บุคคล (ความเชื่อมั่น: 0.07)
  บุคคล (ความเชื่อมั่น: 0.01)
  บุคคล (ความเชื่อมั่น: 0.01)
  บุคคล (ความเชื่อมั่น: 0.01)
  บุคคล (ความเชื่อมั่น: 0.01)
  บุคคล (ความเชื่อมั่น: 0.00)
  บุคคล (ความเชื่อมั่น: 0.00)
  บุคคล (ความเชื่อมั่น: 0.00)
  บุคคล (ความเชื่อมั่น: 0.00)
  ผลลัพธ์ถูกบันทึกใน people.jpg

การวิเคราะห์เสร็จสิ้น