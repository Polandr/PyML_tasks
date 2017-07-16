import pandas as P
data_path = '.\\titanic.csv'
data = P.read_csv(data_path, index_col='PassengerId')

# Calculate male and female passengers
male = female = unknown = 0
for sex in data['Sex']:
    if (sex == 'male'):
        male += 1
    elif (sex == 'female'):
        female += 1
    else:
        unknown+=1
print ('Males: ' + str(male) + ' ,females: ' + str(female) + ' ,unknown sex: ' + str(unknown) +
       ' ,checksum: ' + str(male+female))

# Calculate survive rate
print('Survive rate: %.3f' % (float(data['Survived'].sum())/float(data.index[-1])*100))

# Calculate first class passengers rate
first_class = data['Pclass'].where(data['Pclass'] == 1).count()
all_passengers = data.index[-1]
print('First class rate: %.3f' % (float(first_class)/float(all_passengers)*100))

# Calculate mean and median age
ages = data['Age']
print('Mean age: %.3f ,median age: %.3f' % (ages.mean(), ages.median()))

# Calculate correlation between sisters/brothers/spouses and parents/children
sibsp = data['SibSp']
parch = data['Parch']
print('Pearson correlation between sisters/brothers/spouses and parents/children: %.2f' % (sibsp.corr(parch, method='pearson')))

# Get the most frequent name
female_names = data.where(data['Sex'] == 'female')['Name'].dropna()
#print(female_names)
names_dict = {}
rare_names = []
for name in female_names:
    name = name.split()
    if 'Mrs.' in name:
        first_name = [f_n for f_n in name[name.index('Mrs.'):] if '(' in f_n]
        if not first_name:
            first_name = name[name.index('Mrs.') + 1]
        else:
            first_name = first_name[0]
    elif 'Miss.' in name:
        first_name = name[name.index('Miss.') + 1]
    else:
        rare_names.append(' '.join(name))
    if first_name:
        first_name = first_name.replace('(','')
        first_name = first_name.replace(')','')
        if first_name in names_dict:
            names_dict[first_name] += 1
        else:
            names_dict[first_name] = 1
    else:
        print('Unparseable name ' + name)
        
print('The most frequent name is: ' + sorted(names_dict, key=names_dict.get)[-1])
print('It\'s count is: ' + str(sorted(names_dict.values())[-1]))
print()
print('Rare name type (neither Mrs. nor Miss.):')
for name in rare_names:
    print(name)
