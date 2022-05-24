import numpy
import pandas as pd
from django.shortcuts import render
import numpy as np
from openpyxl.descriptors import Integer


def home(request):
    return render(request, "home.html")


def predict(request):
    return render(request, "predict.html")


def result(request):
    import numpy as np
    df = pd.read_excel(r'C:\Users\user\OneDrive\Desktop\3rdYear\3rdyearproject\projectData\NewData.xlsx')
    df = df[['NAME', 'GENDER', 'YEAR', 'STREAM', 'RESULT', 'ENG', 'DZO', 'PHY', 'CHE', 'MATH', 'BIO']]
    df.drop(['NAME'], axis=1, inplace=True)
    df.drop(df.index[df['RESULT'] == 'ABS'], inplace=True)
    df.dropna(subset=['RESULT'], inplace=True)
    df.YEAR.astype(int)
    df = df.groupby('STREAM')
    df = df.get_group('SCIENCE')
    df.shape
    df.replace(0, np.nan, inplace=True)
    df.isna().sum()

    categorical_data = df.select_dtypes(include=['object'])
    numerical_data = df.select_dtypes(include=['float', 'int64'])
    numerical_data = numerical_data.drop(['YEAR'], axis=1)

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputed_df = imputer.fit_transform(numerical_data)
    numerical_data = pd.DataFrame(data=imputed_df, columns=numerical_data.columns)

    x = numerical_data

    y = categorical_data.drop(['GENDER', 'STREAM'], axis=1)
    from sklearn.preprocessing import LabelEncoder
    LE = LabelEncoder()
    Y = LE.fit_transform(y)
    y = pd.DataFrame(Y)

    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, stratify=y, random_state=5, shuffle=True)

    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(criterion="entropy")
    model.fit(xtrain, ytrain)

    v1 = int(request.GET['eng'])
    v2 = int(request.GET['dzo'])
    v3 = int(request.GET['maths'])
    v4 = int(request.GET['bio'])
    v5 = float(request.GET['phy'])
    v6 = float(request.GET['che'])

    Y = numpy.array([v1, v2, v4, v3, v5, v6])
    np = Y.reshape(-1, 6)
    pred = model.predict(np)

    if pred == 0:
        marks = "PCA"
    else:
        marks = "PCNA"

    return render(request, "predict.html", {"result2": marks})
