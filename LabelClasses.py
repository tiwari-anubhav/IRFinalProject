import pandas as pd
class_positive = ['great', 'awesome', 'amazing', 'love']
class_negative = ['hate', 'worst', 'disappointed', 'disgusted']
class_neutral = ['okay', 'decent', 'fair', 'satisfactory']
positive = []
negative = []
neutral = []
with open("filename.txt", "r") as file:
    for i, line in enumerate(file.readlines()):
        for key in class_positive:
            if key in line.split() and line not in positive and len(positive) < 101:
                positive.append(line)
        for key in class_negative:
            if key in line.split() and line not in negative and not any(item in positive for item in line.split()) and len(negative) < 101:
                negative.append(line)
        for key in class_neutral:
            if key in line.split() and line not in neutral and line not in negative and line not in positive and len(neutral) < 101:
                neutral.append(line)
dict_1 = {"Positive": positive, "Negative": negative, "Neutral": neutral}

df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict_1.items()]))
df.to_csv("classes_labeled.csv")
