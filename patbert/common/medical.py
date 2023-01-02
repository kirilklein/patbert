import numpy as np
import string

def ICD_topic(code):
    """0 is reserved for mapping onto zero vector"""
    if code[:3] >= "A00" and code[:3] <= "B99": # Certain Infectious and Parasitic Diseases
        return 1
    if code[:3] >= "C00" and code[:3] <= "D48": # Neoplasms
        return 2
    if code[:3] >= "D50" and code[:3] <= "D89": # Blood, Blood-Forming Organs, and Certain Disorders Involving the Immune Mechanism
        return 3
    if code[:3] >= "E00" and code[:3] <= "E90": # Endocrine, Nutritional, and Metabolic Diseases, and Immunity Disorders
        return 4
    if code[:3] >= "F00" and code[:3] <= "F99": # Mental, Behavioral, and Neurodevelopmental Disorders 
        return 5
    if code[:3] >= "G00" and code[:3] <= "G99": # Diseases of the Nervous System
        return 6
    if code[:3] >= "H00" and code[:3] <= "H59": # Diseases of the Eye and Adnexa
        return 7
    if code[:3] >= "H60" and code[:3] <= "H95": # Diseases of the Ear and Mastoid Process
        return 8
    if code[:3] >= "I00" and code[:3] <= "I99": # Diseases of the Circulatory System
        return 9
    if code[:3] >= "J00" and code[:3] <= "J99": # Diseases of the Respiratory System
        return 10
    if code[:3] >= "K00" and code[:3] <= "K93": # Diseases of the Digestive System
        return 11
    if code[:3] >= "L00" and code[:3] <= "L99": # Diseases of the Skin and Subcutaneous Tissue
        return 12
    if code[:3] >= "M00" and code[:3] <= "M99": # Diseases of the Musculoskeletal System and Connective Tissue
        return 13
    if code[:3] >= "N00" and code[:3] <= "N99": # Diseases of the Genitourinary System
        return 14
    if code[:3] >= "O00" and code[:3] <= "O99": # Pregnancy, Childbirth, and the Puerperium
        return 15
    if code[:3] >= "P00" and code[:3] <= "P96": # Certain Conditions Originating in the Perinatal Period
        return 16
    if code[:3] >= "Q00" and code[:3] <= "Q99": # Congenital Malformations, Deformations, and Chromosomal Abnormalities
        return 17
    if code[:3] >= "R00" and code[:3] <= "R99": # Symptoms, Signs, and Ill-Defined Conditions
        return 18
    if code[:3] >= "S00" and code[:3] <= "T98": # Injury, Poisoning, and Certain Other Consequences of External Causes
        return 19
    if code[:3] >= "X60" and code[:3] <= "Y09": # External Causes of Injury
        return 20
    if code[:3] >= "Z00" and code[:3] <= "Z99":
        return 21
    else: 
        return 22


# TODO: use subtopics from SKS to further divide the codes
def ATC_topic(code):
    atc_topic_ls = ['A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P', 'R', 'S', 'V']
    atc_topic_dic = {topic:i+1 for i, topic in enumerate(atc_topic_ls)}
    if code[0] in atc_topic_dic:
        return atc_topic_dic[code[0]]
    else:
        return len(atc_topic_ls)+2 #we start at 1, so we need to add 2

def construct_dict(level):
    vocab = {}
    if level==1:
        for a in string.ascii_uppercase:
            for i in range(10):
                for j in range(10):
                    code = a+str(i)+str(j)
                    vocab['D'+code] = ICD_topic(a+str(i)+str(j))
        for a in 'ABCDJGHLMNPRSV':
            for i in range(10):
                for j in range(10):
                    code = a+str(i)+str(j)
                    vocab['M'+code] = ATC_topic(a)
    
    elif level==2:
        #categories
        v1 = construct_dict(1)
        for k in v1:
            # order is important here
            if k.startswith('D'):
                vocab[k] = len(vocab)+1
            elif k.startswith('M'):
                vocab[k] = vocab['D'+k[1:]]
    
    elif level==3:
        v2 = construct_dict(2)
        for k in v2:
            if k.startswith('D'):
                for i in range(10):
                    code = k+str(i)
                    vocab[code] = i+1
        
    return vocab

# Probably not needd
def ICD_topic_arr(codes):
    condlist = [(codes>='A00') & (codes<='B99'),
                (codes>='C00') & (codes<='D48'),
                (codes>='D50') & (codes<='D89'),
                (codes>='E00') & (codes<='E90'),
                (codes>='F00') & (codes<='F99'),
                (codes>='G00') & (codes<='G99'),
                (codes>='H00') & (codes<='H59'),
                (codes>='H60') & (codes<='H95'),
                (codes>='I00') & (codes<='I99'),
                (codes>='J00') & (codes<='J99'),
                (codes>='K00') & (codes<='K93'),
                (codes>='L00') & (codes<='L99'),
                (codes>='M00') & (codes<='M99'),
                (codes>='N00') & (codes<='N99'),
                (codes>='O00') & (codes<='O99'),
                (codes>='P00') & (codes<='P96'),
                (codes>='Q00') & (codes<='Q99'),
                (codes>='R00') & (codes<='R99'),
                (codes>='S00') & (codes<='T98'),
                (codes>='X60') & (codes<='Y09'),
                (codes>='Z00') & (codes<='Z99')]
    choicelist = ['Dg'+str(i) for i in range(len(condlist))]
    choicelist.append('DUNK')
    return np.select(condlist, choicelist, default=choicelist[-1]), choicelist
