# IR-Remote-Controller-with-Hand-Gestures
# Summary
The aim of this project, is to control a IR remote controller with hand gestures. To achieve this goal, we have done image filtering to distinguish hand from the background view. Then closed contour of hand is extracted from the filtered image. Maximum points in y-axis on the contour are detected. If one of these points is further from the center of the contour than a threshold value, then it will be counted as a finger. Depends on the position of fingers, or another defined movements will be turned into IR signals with help of LIRC library on Raspberry Pi 3B board.

This project is done by Burak Yakup Çakar & Serhat Yusuf Öztaşkın
