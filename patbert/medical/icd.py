def ICD_topic(code):
    if code[:3] >= "A00" and code[:3] <= "B99":
        return 1
    if code[:3] >= "C00" and code[:3] <= "D49":
        return 2
    if code[:3] >= "D50" and code[:3] <= "D89":
        return 3
    if code[:3] >= "E00" and code[:3] <= "E90":
        return 4
    if code[:3] >= "F01" and code[:3] <= "F99":
        return 5
    if code[:3] >= "G00" and code[:3] <= "G99":
        return 6
    if code[:3] >= "H00" and code[:3] <= "H59":
        return 7
    if code[:3] >= "H60" and code[:3] <= "H95":
        return 8
    if code[:3] >= "I00" and code[:3] <= "I99":
        return 9
    if code[:3] >= "J00" and code[:3] <= "J99":
        return 10
    if code[:3] >= "K00" and code[:3] <= "K93":
        return 11
    if code[:3] >= "L00" and code[:3] <= "L99":
        return 12
    if code[:3] >= "M00" and code[:3] <= "M99":
        return 13
    if code[:3] >= "N00" and code[:3] <= "N99":
        return 14
    if code[:3] >= "O00" and code[:3] <= "O99":
        return 15
    if code[:3] >= "P00" and code[:3] <= "P96":
        return 16
    if code[:3] >= "Q00" and code[:3] <= "Q99":
        return 17
    if code[:3] >= "R00" and code[:3] <= "R99":
        return 18
    if code[:3] >= "S00" and code[:3] <= "T98":
        return 19
    if code[:3] >= "V01" and code[:3] <= "Y98":
        return 20
    if code[:3] >= "Z00" and code[:3] <= "Z99":
        return 21
    else: 
        return -1


