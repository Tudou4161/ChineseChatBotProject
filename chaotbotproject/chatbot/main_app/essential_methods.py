#response를 위한 딥러닝 코드!
# 태그 단어
PAD = "<PADDING>"   # 패딩
STA = "<START>"     # 시작
END = "<END>"       # 끝
OOV = "<OOV>"       # 없는 단어(Out of Vocabulary)

# 태그 인덱스
PAD_INDEX = 0
STA_INDEX = 1
END_INDEX = 2
OOV_INDEX = 3
# Create your views here.
# 예측을 위한 입력 생성
max_sequences = 30

# 데이터 타입
ENCODER_INPUT  = 0
DECODER_INPUT  = 1
DECODER_TARGET = 2

# 정규 표현식 필터
RE_FILTER = re.compile("[.,!?\"':;~()]")

# 인덱스를 문장으로 변환
# 챗봇 데이터 로드
csv_path = r"C:\Users\WIN10\Desktop\UCCProject\ChineseChatBotProject\chaotbotproject\chatbot\model\Total_dataset.xlsx"

chatbot_data = pd.read_excel(csv_path)
question, answer = list(chatbot_data['Q']), list(chatbot_data['A'])

# 형태소분석 함수
def pos_tag(sentences):
    #중국어 형태소 분석기 설정
    #j.enable_paddle()
    
    # 문장 품사 변수 초기화
    sentences_pos = []
    
    # 모든 문장 반복
    for sentence in sentences:
        # 특수기호 제거
        sentence = re.sub(RE_FILTER, "", sentence)
        
        # 배열인 형태소분석의 출력을 띄어쓰기로 구분하여 붙임
        #sentence = " ".join(tagger.morphs(sentence))
        sentences_pos.append(jieba.cut(sentence))
        print(sentences)
        
    return sentences_pos

question = pos_tag(question)
answer = pos_tag(answer)

sentences = []
sentences.extend(question)
sentences.extend(answer)

words = []

# 단어들의 배열 생성
for sentence in sentences:
    for word in sentence:
        words.append(word)

# 길이가 0인 단어는 삭제
words = [word for word in words if len(word) > 0]

# 중복된 단어 삭제
words = list(set(words))
print(len(words))
# 제일 앞에 태그 단어 삽입
words[:0] = [PAD, STA, END, OOV]
print(words)

word_to_index = {word: index for index, word in enumerate(words)}
index_to_word = {index: word for index, word in enumerate(words)}

print(index_to_word)

#인덱스를 단어조합의 문 장으로 변환하는 함수
def convert_index_to_text(indexs, vocabulary): 
    
    sentence = ''
    
    # 모든 문장에 대해서 반복
    for index in indexs:
        if index == END_INDEX:
            # 종료 인덱스면 중지
            break;
        if vocabulary.get(index) is not None:
            # 사전에 있는 인덱스면 해당 단어를 추가
            sentence += vocabulary[index]
        else:
            # 사전에 없는 인덱스면 OOV 단어를 추가
            sentence.extend([vocabulary[OOV_INDEX]])
            
        # 빈칸 추가
        sentence += ' '

    return sentence

# 문장을 단어별로 분리해서 인덱스로 변환하는 함수
def convert_text_to_index(sentences, vocabulary, type): 
    
    sentences_index = []
    
    # 모든 문장에 대해서 반복
    for sentence in sentences:
        sentence_index = []
        
        # 디코더 입력일 경우 맨 앞에 START 태그 추가
        if type == DECODER_INPUT:
            sentence_index.extend([vocabulary[STA]])
        
        # 문장의 단어들을 띄어쓰기로 분리
        for word in sentence:
            if vocabulary.get(word) is not None:
                # 사전에 있는 단어면 해당 인덱스를 추가
                sentence_index.extend([vocabulary[word]])
            else:
                # 사전에 없는 단어면 OOV 인덱스를 추가
                sentence_index.extend([vocabulary[OOV]])

        # 최대 길이 검사
        if type == DECODER_TARGET:
            # 디코더 목표일 경우 맨 뒤에 END 태그 추가
            if len(sentence_index) >= max_sequences:
                sentence_index = sentence_index[:max_sequences-1] + [vocabulary[END]]
            else:
                sentence_index += [vocabulary[END]]
        else:
            if len(sentence_index) > max_sequences:
                sentence_index = sentence_index[:max_sequences]
            
        # 최대 길이에 없는 공간은 패딩 인덱스로 채움
        sentence_index += (max_sequences - len(sentence_index)) * [vocabulary[PAD]]
        
        # 문장의 인덱스 배열을 추가
        sentences_index.append(sentence_index)
        print(sentences_index)
    return np.asarray(sentences_index)


#질문 
def make_predict_input(sentence):

    sentences = []
    sentences.append(sentence)
    sentences = pos_tag(sentences)
    input_seq = convert_text_to_index(sentences, word_to_index, ENCODER_INPUT)
    
    return input_seq


#대답
def generate_text(input_seq):

    encoder_model = load_model(r"C:\Users\WIN10\Desktop\UCCProject\ChineseChatBotProject\chaotbotproject\chatbot\model\encoder_model3.h5")    
    # 입력을 인코더에 넣어 마지막 상태 구함
    states = encoder_model.predict(input_seq)

    # 목표 시퀀스 초기화
    target_seq = np.zeros((1, 1))
    
    # 목표 시퀀스의 첫 번째에 <START> 태그 추가
    target_seq[0, 0] = STA_INDEX
    
    # 인덱스 초기화
    indexs = []
    
    decoder_model = load_model(r"C:\Users\WIN10\Desktop\UCCProject\ChineseChatBotProject\chaotbotproject\chatbot\model\decoder_model3.h5")

    while 1:
        # 디코더로 현재 타임 스텝 출력 구함
        # 처음에는 인코더 상태를, 다음부터 이전 디코더 상태로 초기화
        decoder_outputs, state_h, state_c = decoder_model.predict(
                                                [target_seq] + states)

        # 결과의 원핫인코딩 형식을 인덱스로 변환
        index = np.argmax(decoder_outputs[0, 0, :])
        indexs.append(index)
        
        # 종료 검사
        if index == END_INDEX or len(indexs) >= max_sequences:
            break

        # 목표 시퀀스를 바로 이전의 출력으로 설정
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = index
        
        # 디코더의 이전 상태를 다음 디코더 예측에 사용
        states = [state_h, state_c]

    # 인덱스를 문장으로 변환
    sentence = convert_index_to_text(indexs, index_to_word)    
    return sentence
