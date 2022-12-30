def ICD_topic(code):
    if code[:3] >= "A00" and code[:3] <= "B99": # Certain Infectious and Parasitic Diseases
        return 0
    if code[:3] >= "C00" and code[:3] <= "D48": # Neoplasms
        return 1
    if code[:3] >= "D50" and code[:3] <= "D89": # Blood, Blood-Forming Organs, and Certain Disorders Involving the Immune Mechanism
        return 2
    if code[:3] >= "E00" and code[:3] <= "E90": # Endocrine, Nutritional, and Metabolic Diseases, and Immunity Disorders
        return 3
    if code[:3] >= "F00" and code[:3] <= "F99": # Mental, Behavioral, and Neurodevelopmental Disorders 
        return 4
    if code[:3] >= "G00" and code[:3] <= "G99": # Diseases of the Nervous System
        return 5
    if code[:3] >= "H00" and code[:3] <= "H59": # Diseases of the Eye and Adnexa
        return 6
    if code[:3] >= "H60" and code[:3] <= "H95": # Diseases of the Ear and Mastoid Process
        return 7
    if code[:3] >= "I00" and code[:3] <= "I99": # Diseases of the Circulatory System
        return 8
    if code[:3] >= "J00" and code[:3] <= "J99": # Diseases of the Respiratory System
        return 9
    if code[:3] >= "K00" and code[:3] <= "K93": # Diseases of the Digestive System
        return 10
    if code[:3] >= "L00" and code[:3] <= "L99": # Diseases of the Skin and Subcutaneous Tissue
        return 11
    if code[:3] >= "M00" and code[:3] <= "M99": # Diseases of the Musculoskeletal System and Connective Tissue
        return 12
    if code[:3] >= "N00" and code[:3] <= "N99": # Diseases of the Genitourinary System
        return 13
    if code[:3] >= "O00" and code[:3] <= "O99": # Pregnancy, Childbirth, and the Puerperium
        return 14
    if code[:3] >= "P00" and code[:3] <= "P96": # Certain Conditions Originating in the Perinatal Period
        return 15
    if code[:3] >= "Q00" and code[:3] <= "Q99": # Congenital Malformations, Deformations, and Chromosomal Abnormalities
        return 16
    if code[:3] >= "R00" and code[:3] <= "R99": # Symptoms, Signs, and Ill-Defined Conditions
        return 17
    if code[:3] >= "S00" and code[:3] <= "T98": # Injury, Poisoning, and Certain Other Consequences of External Causes
        return 18
    if code[:3] >= "X60" and code[:3] <= "Y09": # External Causes of Injury
        return 19
    if code[:3] >= "Z00" and code[:3] <= "Z99":
        return 20
    else: 
        return 21

# TODO: use subtopics from SKS to further divide the codes
def ATC_topic(code):
    atc_topic_ls = ['A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P', 'R', 'S', 'V']
    atc_topic_dic = {topic:i for i, topic in enumerate(atc_topic_ls)}
    if code[0] in atc_topic_dic:
        return atc_topic_dic[code[0]]
    else:
        return 15