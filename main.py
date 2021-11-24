from lib.meter_ocr import MeterOcr;
import logging

logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    ocr = MeterOcr('samples/meter7.jpg')
    result = ocr.getDigits()

