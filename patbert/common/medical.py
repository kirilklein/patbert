import string
import pickle as pkl



"""  ("UA","UA"), # Abdominal Circumference [cm]
        ("UB","UB"), # Orificiums opening into [cm]
        ("UH","UH"), # Head circumference [cm]
        ("UP","UP"), # Fetus
        ("UT","UT"), # Mother
        ("V00","V99"), # Weight of the placenta [g]
        ("VA","VA"), # Undisclosed
        ("VRA","VRA"), # regarding weight, height 
        ("VRB","VRB"), # tobacco and alcohol
        ("VRK","VRK"), # interoperative bleeding"""

"""
def ICD_topic(code):
    
    c = code[:3]
    if "A00" <= c <= "B99": # Certain Infectious and Parasitic Diseases
        return 1
    elif c >= "C00" and c <= "D48": 
        return 2
    elif c >= "D50" and c <= "D89": # Blood, Blood-Forming Organs, and Certain Disorders Involving the Immune Mechanism
        return 3
    elif c >= "E00" and c <= "E90": # Endocrine, Nutritional, and Metabolic Diseases, and Immunity Disorders
        return 4
    elif c >= "F00" and c <= "F99": # Mental, Behavioral, and Neurodevelopmental Disorders 
        return 5
    elif c >= "G00" and c <= "G99": # Diseases of the Nervous System
        return 6
    elif c >= "H00" and c <= "H59": # Diseases of the Eye and Adnexa
        return 7
    elif c >= "H60" and c <= "H95": # Diseases of the Ear and Mastoid Process
        return 8
    elif c >= "I00" and c <= "I99": # Diseases of the Circulatory System
        return 9
    elif c >= "J00" and c <= "J99": # Diseases of the Respiratory System
        return 10
    elif c >= "K00" and c <= "K93": # Diseases of the Digestive System
        return 11
    elif c >= "L00" and c <= "L99": # Diseases of the Skin and Subcutaneous Tissue
        return 12
    elif c >= "M00" and c <= "M99": # Diseases of the Musculoskeletal System and Connective Tissue
        return 13
    elif c >= "N00" and c <= "N99": # Diseases of the Genitourinary System
        return 14
    elif c >= "O00" and c <= "O99": # Pregnancy, Childbirth, and the Puerperium
        return 15
    elif c >= "P00" and c <= "P96": # Certain Conditions Originating in the Perinatal Period
        return 16
    elif c >= "Q00" and c <= "Q99": # Congenital Malformations, Deformations, and Chromosomal Abnormalities
        return 17
    elif c >= "R00" and c <= "R99": # Symptoms, Signs, and Ill-Defined Conditions
        return 18
    elif c >= "S00" and c <= "T98": # Injury, Poisoning, and Certain Other Consequences of External Causes
        return 19
    elif c >= "X60" and c <= "Y09": # External Causes of Injury
        return 20
    elif c >= "Z00" and c <= "Z99": #
        return 21
    elif c[0] == 'U': #special codes (childbirth and pregnancy)
        return 22
    else: 
        return 23
"""

def sks_codes_to_list():
    """Convert the SKScomplete list of codes to a list of codes in pickles format."""
    codes = []
    with open("..\\..\\data\\medical\\SKScomplete.txt") as f:
        for line in f:
            codes.append(line.split(' ')[0])
    codes = set(codes)
    with open("..\\..\\data\\medical\\SKScodes.pkl", "wb") as f:
        pkl.dump(codes, f)

class SKS_CODES():
    """get a list of SKS codes of a certain type
    We will construct a dictionary for Lab Tests on the fly"""
    def __init__(self):
        with open("..\\..\\data\\medical\\SKScodes.pkl", "rb") as f:
            self.codes = pkl.load(f)
        self.vocabs = []

    def __call__(self):
        """return vocab dics"""
        for lvl in range(4):
            self.vocabs.append(self.construct_vocab_dic(lvl))

    def get_codes(self, signature, min_len=2):
        codes =[c.strip(signature) for c in self.codes if c.startswith(signature)]
        return [c for c in codes if len(c)>=min_len]
        
    def get_icd(self):
        return self.get_codes('dia', min_len=4)
    def get_atc(self):
        codes = self.get_codes('atc', min_len=4)
        codes[codes.index('N05CC10')] = 'MZ99' # thalidomid, wrongly generated code will be assigned a special token
        return codes
    def get_adm(self):
        print("Not finished yet!")
        return self.get_codes('adm')
    def get_operations(self):
        print("Not finished yet!")
        return self.get_codes('opr')
    def get_procedures(self):
        print("Not finished yet!")
        return self.get_codes('pro')
    def get_special_codes(self):
        print("Not finished yet!")
        return self.get_codes('til')
    def get_ext_injuries(self):
        print("Not finished yet!")
        return self.get_codes('uly')
    def get_studies(self):
        """MR scans, CT scans, etc."""
        return self.get_codes('und')

    def construct_vocab_dic(self, level):
        """construct a dictionary of codes and their topics"""
        vocab = {}
        assert level<=5, "Level must be between 1 and 5"
        icd_codes = self.get_icd()
        atc_codes = self.get_atc()
        if level==1:
            """Topic level e.g. A00-B99 (Neoplasms), 
            C00-D48 (Diseases of the blood and blood-forming organs), etc."""
            for code in icd_codes:
                vocab[code] = self.ICD_topic(code)
            for code in atc_codes:
                vocab[code] = self.ATC_topic(code)
            # TODO: add other codes
        else:
            # Looks good so far
            vocab = self.enumerate_codes_lvl(icd_codes, vocab, level)
            vocab = self.enumerate_codes_lvl(atc_codes, vocab, level)
        return vocab
    
    
    
    def get_temp_vocab_atc_cat(self, codes):
        pass

    @staticmethod
    def insert_voc(code, voc):
        """Insert a code into the vocabulary"""
        if code not in voc:
            voc[code] = len(voc)+1
        return voc
    

    def enumerate_codes_lvl(self, codes, vocab, lvl):
        """Uses the temporary vocabulary to assign a category to each code."""
        if lvl==2:
            if codes[0].startswith('D'):
                temp_vocab = self.get_temp_vocab_icd(codes, lvl)
            elif codes[0].startswith('M'):
                pass
            else:
                print(f"Code type starting with {code[0]} not implemented yet")

            for code in codes:
                if code.startswith('D'):    
                    if code.startswith('DU') or code.startswith('DV'):
                        # special codes
                        #TODO: have a closer look at pregnancy weeks!
                        if code[2].isdigit():
                            # special code followed by two digits  
                            vocab[code] =  temp_vocab[code[:2]]
                        elif code[2]=='R':
                            vocab[code] = temp_vocab[code[:4]]
                        else:
                            vocab[code] =  temp_vocab[code[:3]]
                    else:
                        vocab[code] =  temp_vocab[code[:4]]
        elif lvl==3:
            pass
        return vocab

    
    def get_temp_vocab_icd(self, codes, lvl):
        """Construct a temporary vocabulary for categories for icd codes"""
        temp_vocab = {}                
        special_codes_u = ['DUA', 'DUB', 'DUH', 'DUP', 'DUT'] # different birth-related codes
        special_codes_v = ['DVA', 'DVRA', 'DVRB', 'DVRK'] # placenta weight, height weight ...
        special_codes = special_codes_u + special_codes_v
        if lvl==2:
            for code in codes:
                if code.startswith('DU') or code.startswith('DV'):
                    special_code_bool = [code.startswith(s) for s in special_codes]
                    if any(special_code_bool):
                        key = special_codes[special_code_bool.index(True)]
                        temp_vocab = self.insert_voc(key, temp_vocab)
                        continue 
                    if code[3].isdigit(): # duration of pregancy DUwwDdd
                        temp_vocab = self.insert_voc(code[:2], temp_vocab)
                        continue 
                else: 
                    temp_vocab = self.insert_voc(code[:4], temp_vocab)
        elif lvl==3:
            pass
        return temp_vocab

    def enumerate_codes_subcategory(self, codes, vocab):
        an_vocab = self.alphanumeric_vocab()
        for code in codes:
            if len(code)<5:
                vocab[code] = 0
            else:
                vocab[code] = an_vocab[code[4]]
        return vocab

    @staticmethod
    def two_digit_vocab():
        """Construct a vocabulary for two digit codes"""
        vocab = {}
        for i in range(10):
            for j in range(10):
                vocab[str(i)+str(j)] = i*10+j+1
        return vocab
    @staticmethod
    def alphanumeric_vocab():
        alphanumeric = string.ascii_uppercase + string.digits
        vocab = {a:i+1 for i, a in enumerate(alphanumeric)} 
        # first value is reserved for 0
        return vocab

    @staticmethod
    def ATC_topic(code):
        assert code[0] == 'M', f"ATC code must start with 'M, code: {code}'"
        atc_topic_ls = ['A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P', 'R', 'S', 'V']
        atc_topic_dic = {topic:(i+1) for i, topic in enumerate(atc_topic_ls)}
        if code[1] in atc_topic_dic:
            return atc_topic_dic[code[1]]
        else:
            return len(atc_topic_ls)+2 #we start at 1, so we need to add 2
    @staticmethod
    def ICD_topic(code):
        assert code[0] == 'D', f"ICD code must start with 'D', code: {code}"
        options = [
            ("A00","B99"), # Certain Infectious and Parasitic Diseases
            ("C00","D48"), # Neoplasms
            ("D50","D89"), # Blood, Blood-Forming Organs, and Certain Disorders Involving the Immune Mechanism
            ("E00","E90"), # Endocrine, Nutritional, and Metabolic Diseases, and Immunity Disorders
            ("F00","F99"), # Mental, Behavioral, and Neurodevelopmental Disorders
            ("G00","G99"), # Diseases of the Nervous System
            ("H00","H59"), # Diseases of the Eye and Adnexa
            ("H60","H95"), # Diseases of the Ear and Mastoid Process
            ("I00","I99"), # Diseases of the Circulatory System
            ("J00","J99"), # Diseases of the Respiratory System
            ("K00","K93"), # Diseases of the Digestive System
            ("L00","L99"), # Diseases of the Skin and Subcutaneous Tissue
            ("M00","M99"), # Diseases of the Musculoskeletal System and Connective Tissue
            ("N00","N99"), # Diseases of the Genitourinary System
            ("O00","O99"), # Pregnancy, Childbirth, and the Puerperium
            ("P00","P96"), # Certain Conditions Originating in the Perinatal Period
            ("Q00","Q99"), # Congenital Malformations, Deformations, and Chromosomal Abnormalities
            ("R00","R99"), # Symptoms, Signs, and Ill-Defined Conditions
            ("S00","T98"), # Injury, Poisoning, and Certain Other Consequences of External Causes
            ("X60","Y09"), # External Causes of Injury
            ("Z00","Z99"), # Factors Influencing Health Status and Contact with Health Services
        ]   
        for i, option in enumerate(options):
            if option[0] <= code[1:4] <= option[1]:
                return i+1
            elif code.startswith("DU"): # special codes (childbirth and pregnancy)
                return len(options)+2
            elif code.startswith("DV"): #weight, height and various other codes
                return len(options)+3
        return len(options)+4





def construct_dict(level):
    """construct a dictionary of codes and their topics"""
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


