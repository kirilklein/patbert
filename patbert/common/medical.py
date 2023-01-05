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
        ("VRK","VRK"), # interoperative bleeding
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
        codes = icd_codes + atc_codes
        if level==0:
            special_tokens = ['<CLS>', '<PAD>', '<SEP>', '<MASK>', '<UNK>', 
                        '<MALE>', '<FEMALE>', '<BIRTHYEAR>', '<BIRTHMONTH>',
                        'adm','dia', 'atc','opr', 'pro', 'til', 'uly', 'und', 'lab']
            vocab = {token:idx for idx, token in enumerate(special_tokens)}
        if level==1:
            """Topic level e.g. A00-B99 (Neoplasms), 
            C00-D48 (Diseases of the blood and blood-forming organs), etc."""
            for code in codes:
                vocab[code] = self.topic(code)
        else:
            # Looks good so far
            vocab = self.enumerate_codes_lvl(icd_codes, vocab, level)
            vocab = self.enumerate_codes_lvl(atc_codes, vocab, level)
        return vocab
    

    @staticmethod
    def insert_voc(code, voc):
        """Insert a code into the vocabulary"""
        if code not in voc:
            voc[code] = len(voc)
        return voc
    @staticmethod
    def same_type(codes):
        # Get the first character of the first string
        first_char = codes[0][0]
        # Iterate over the strings and compare the first character of each string to the first character of the first string
        for code in codes:
            if code[0] != first_char:
                return False
        return True

    def enumerate_codes_lvl(self, codes, vocab, lvl):
        """Uses the temporary vocabulary to assign a category to each code."""
        assert self.same_type(codes), "Codes must be of the same type"
        if codes[0].startswith('D'): # TODO: this might be improved, should not accept mixed codes
                temp_vocab = self.get_temp_vocab_icd(codes, lvl)
        elif codes[0].startswith('M'):
            pass
        else:
            print(f"Code type starting with {code[0]} not implemented yet")

        for code in codes:
            if code.startswith('D'):    
                if code.startswith(('DU', 'DV')):
                    # special codes
                    #TODO: have a closer look at pregnancy weeks!
                    if code[2].isdigit():
                        # special code followed by two digits 
                        if lvl==2: 
                            vocab[code] =  temp_vocab[code[:2]]
                        else:
                            vocab[code] = 0 # we fill all level below with zero

                    elif code.startswith(('DUA', 'DUB', 'DUH')):
                        if lvl==2:
                            vocab[code] = temp_vocab[code[:3]]
                        else: #DVRA, DVRB, DVRK
                            vocab[code] = 0 # we fill all level below with zeros
                    elif code=='DVRK01':
                        if lvl==2:
                            vocab[code] = temp_vocab[code]
                        else:
                            vocab[code] = 0
                    elif code.startswith(('DUP', 'DUT', 'DVA')):
                        if lvl==2:
                            vocab[code] = temp_vocab[code[:3]]
                        elif lvl==3:
                            if code[3].isdigit():
                                vocab[code] = temp_vocab[str(int(code[3:]))] # digits
                            else:
                                vocab[code] = temp_vocab[code[3:]]
                        else:
                            vocab[code] = 0
                    elif code.startswith(('DVRA', 'DVRB')):
                        if lvl==2:
                            vocab[code] = temp_vocab[code[:4]]
                        elif lvl==3:
                            if  self.all_digits(code[4:]):
                                vocab[code] = temp_vocab[str(int(code[4:]))] # digits
                            else:
                                vocab[code] = temp_vocab[code[4:]] #TODO: check this
                        else:
                            vocab[code] = 0
                    else:
                        vocab[code] =  temp_vocab[code[:4]]
                else:
                    if lvl==2:
                        vocab[code] =  temp_vocab[code[:4]]
                    else:
                        if (lvl+2)<=len(code): 
                            vocab[code] = temp_vocab[code[lvl+1]]
                        else:
                            vocab[code] = 0

        return vocab
    @staticmethod
    def all_digits(codes):
        """Check if a string only contains digits"""
        return all([c.isdigit() for c in codes])
         
    def get_temp_vocab_icd(self, codes, lvl):
        """Construct a temporary vocabulary for categories for icd codes"""
        temp_vocab = {'<ZERO>':0,'<UNK>':1}                
        special_codes_u = ['DUA', 'DUB', 'DUH', 'DUP', 'DUT'] # different birth-related codes
        special_codes_v = ['DVA', 'DVRA', 'DVRB', 'DVRK01'] # placenta weight, height weight ...
        special_codes = special_codes_u + special_codes_v
        for code in codes:
            if code.startswith('DU') or code.startswith('DV'):
                # special codes
                special_code_bool = [code.startswith(s) for s in special_codes]
                if any(special_code_bool):
                    key = special_codes[special_code_bool.index(True)]
                    if lvl==2:
                        temp_vocab = self.insert_voc(key, temp_vocab) 
                    # elif lvl==3:
                        # zero_codes = [key.startswith(s) for s in ['DUA', 'DUB', 'DUH', 'DVRK01']]
                        # if any(zero_codes):
                            # temp_vocab[code] = 0
                        # elif code.startswith():
                            # temp_vocab = self.insert_voc(code[:4], temp_vocab)

                elif code[3].isdigit(): # duration of pregancy DUwwDdd or size of placenta 
                    if lvl==2:
                        temp_vocab = self.insert_voc(code[:2], temp_vocab)
                    else:
                        pass
                else:
                    if lvl==2:
                        temp_vocab = self.insert_voc(code[:3], temp_vocab)
            else: 
                if lvl==2:
                    temp_vocab = self.insert_voc(code[:4], temp_vocab)
                    
        if lvl>=3:
            for i in range(10):
                temp_vocab[str(i)] = len(temp_vocab)
            for i in range(10):
                for j in range(10):
                    temp_vocab[str(i)+str(j)] = len(temp_vocab)
            for a in string.ascii_uppercase:
                temp_vocab[a] = len(temp_vocab)
            temp_vocab['XX'] = len(temp_vocab)
            temp_vocab['02A'] = len(temp_vocab)
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
        vocab = {'<ZERO>':0, '<UNK>':1}
        for i in range(10):
            for j in range(10):
                vocab[str(i)+str(j)] = i*10+j+2
        return vocab
    @staticmethod
    def alphanumeric_vocab():
        vocab = {'<ZERO>':0, '<UNK>':1}
        alphanumeric = string.ascii_uppercase + string.digits
        vocab = {a:i+1 for i, a in enumerate(alphanumeric)} 
        # first value is reserved for 0
        return vocab
    def topic(self, code):
        if code.startswith('M'):
            return self.ATC_topic(code)
        elif code.startswith('D'):
            return self.ICD_topic(code)
        else:
            print(f"Code type starting with {code[0]} not implemented yet")
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






