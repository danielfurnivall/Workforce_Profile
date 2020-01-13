import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib import colors
import datetime

from tkinter.filedialog import askopenfilename
plt.rcParams['savefig.transparent']=True
from decimal import Decimal

#This reads in the extract data.
#TODO change this to a prompt to get the extract data instead of this

data_file = askopenfilename(initialdir='//ntserver5/generalDB/WorkforceDB/MFT/Workforce Profiles',
                           filetypes=(("CSV File", "*.csv"), ("All Files", "*.*")),
                           title="Choose the relevant staff extract."
                           )
df = pd.read_csv(data_file)
print("Pre-merge length: " + str(len(df)))

#This reads the retirement file - created every month using "retirement lookup creator.py" in
#"RetirementProjections" pycharm project
#TODO change to prompt
retire_data = askopenfilename(initialdir='//ntserver5/generalDB/WorkforceDB/Retirement Vulnerability',
                           filetypes=(("CSV File", "*.csv"), ("All Files", "*.*")),
                           title="Choose the relevant recruitment file."
                           )
retirestats = pd.read_csv(retire_data)


#This adds retirement projection figures to each employee in the extract
df = df.merge(retirestats[['Pay_Number','Over50', 'This year','1-2 years', '2-3 years', '3-5 years', 'time_to_retire',
                           ]], on='Pay_Number', how='left')

print("Post-merge length: " + str(len(df))) #Check to see that we have the same number of people post merge

cost_centres = df['Cost_Centre'].unique().tolist() #captures cost centres in a list

print(pd.pivot_table(df, values='WTE', index='Cost_Centre', aggfunc = np.sum))

#Below code builds dept and subdir lookups based on cost centre (can map to columns as desired)
lookups = df[['Cost_Centre', 'department', 'Sub-Directorate 1']]
deptlookup = {row[0]: row[1] for row in lookups.values}
depts = df['department'].unique().tolist()
subdirlookup = {row[0]: row[2] for row in lookups.values}

#Build list of MHOs
mho=pd.read_excel('W:/MFT/Workforce Profiles/MHO_Status.xls', date_format='%Y-%m-%d', skiprows=3)
mhos = mho['Pay Number'].tolist()
mhos = df[df['Pay_Number'].isin(mhos)]



#The below code will be used to edit pay bands to new pay band structure
#TODO implement
df['Pay_Band'] = df['Pay_Band'].map({'8A':'A', '8B':'B', '8C':'C', '8D':'D', '2':'2',
                                     '3':'3', '4':'4', '5':'5','6':'6', '7':'7',
                                     '9':'9', 'Medical and Dental':'Medical and Dental', 'Non AFC':'Non AFC'})

bins = [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100]
labels = ['16-24', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65+']
df['Age Group'] = pd.cut(df['Age'], bins = bins, labels = labels, right = False)

#Create registered/unregistered split
df['Reg/Unreg'] = np.where(df['Pay_Band'].isin(['2', '3', '4']), 'Unregistered', 'Registered')


#The below code creates a 'Years of service' field by first converting all dates to datetimes
#then taking the minimum, then taking away from current date
df['Date_Superannuation_Started'] = pd.to_datetime(df['Date_Superannuation_Started'], dayfirst= True)
df['Date_Started'] = pd.to_datetime(df['Date_Started'], dayfirst=True)
df['Date_To_Grade'] = pd.to_datetime(df['Date_To_Grade'], dayfirst=True)
df['EarliestDate'] = df[['Date_Superannuation_Started','Date_Started','Date_To_Grade']].min(axis=1)
df['Years_Of_Service'] = (pd.datetime.now() - df['EarliestDate']).dt.days / 365.25


#The following creates a numeric band column (as opposed to string)
df['BandNumeric'] = df['Pay_Band'].map({'A': 8, 'B': 8.25, 'C': 8.5, 'D': 8.75, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
                                        '7': 7,'9': 9, 'Medical and Dental': 9, 'Non AFC': 9})
print(df[df['Job_Family'] == 'Allied Health Profession']['BandNumeric'].value_counts())

print(df['Pay_Band'].value_counts())

#Read in area balanced scorecard data
abs_data = askopenfilename(initialdir='//ntserver5/generalDB/WorkforceDB/Workforce Monthly Reports/Monthly_Reports',
                           filetypes=(("Excel File", "*.xls"), ("All Files", "*.*")),
                           title="Choose the relevant absence file."
                           )
abs = pd.read_excel(abs_data, sheet_name='Department')
# abs = pd.read_csv('W:/MFT/Workforce Profiles/absdata.csv')
abs = abs[abs['Department'].isin(depts)] #limit absence data to only include relevant depts

#Create lookups for absence
abslookup_dept = abs[['Department', 'WTE', 'Annual WTE', 'Maternity WTE', 'Total Abs WTE']]
wtelookup_dept = {row[0]: row[1] for row in abslookup_dept.values}
annWTE_dept = {row[0]: row[2] for row in abslookup_dept.values}
matWTE_dept = {row[0]: row[3] for row in abslookup_dept.values}
totalAbs_dept = {row[0]: row[4] for row in abslookup_dept.values}

abslookup_sd = abs[['Sub-Directorate 1', 'WTE', 'Annual WTE', 'Maternity WTE', 'Total Abs WTE']]
wtelookup_sd = {row[0]: row[1] for row in abslookup_sd.values}
annWTE_sd = {row[0]: row[2] for row in abslookup_sd.values}
matWTE_sd = {row[0]: row[3] for row in abslookup_sd.values}
totalAbs_sd = {row[0]: row[4] for row in abslookup_sd.values}


#Create groups for projected retirement
labels = ['Within a year', '1-2 years', '2-3 years', '3-5 years', '5-10 years', '>10 years']
bins = [-100, 1, 2, 3, 5, 10, 100]
df['ProjRet'] = pd.cut(df['time_to_retire'], bins=bins, labels=labels, right=False)



leavers = pd.read_excel('W:/Starters & Leavers/Starters and Leavers - Apr 16 - Present.xlsx',
                   sheet_name='Leavers - Apr-16 - Present')
leavers = leavers[(leavers['Year'] == '2018/19') & (leavers['Cost Centre'].isin(cost_centres))]
leavers['Reg/Unreg'] = np.where(leavers['Pay_Band'].isin(['2', '3', '4']), 'Unregistered', 'Registered')
print(len(leavers))
turnover = len(leavers) / len(df) * 100
print(turnover)

#Read in bank data
path = 'W:/Bank Data v2/'
grades = eval(open(path + 'Grades_dic.txt').read())
agencies = eval(open(path + 'Agency_dic.txt').read())
bank_data = pd.read_excel('W:/MFT/Workforce Profiles/bankdata-Sept.xlsm')
bank_data['Department'] = bank_data['Cost Centre'].map(deptlookup)
bank_data['Sub-Directorate 1'] = bank_data['Cost Centre'].map(subdirlookup)
bank_data['Grade'] = bank_data['Request Grade'].map(grades)

df['PayBand3'] = np.where(df['Job_Family'] == 'Medical and Dental', df['Sub_Job_Family'], df['Pay_Band'])


#the below creates specific areas of scope for older peoples'

def econcans():
    north_ecans =['G7089260', 'G0444472', 'G9322116', 'G3864235']
    north_econs = ['G9180036', 'G3906655']
    clyde_ecans = ['C1087770']
    clyde_econs = ['G9865070','G0000824']
    clyde_ecats = ['G9876060', 'G9885030', 'G9868067']
    clyde_fps = ['G9366717', 'C9549560']
    all_econs = []
    all_econs.extend(north_econs+clyde_econs)
    all_econs = df[df['Pay_Number'].isin(all_econs)]
    econs_age = np.average(all_econs['Age'])
    econs_wte = round(np.sum(all_econs['WTE']), 1)
    econs_yos = round(np.average(all_econs['Years_Of_Service']), 1)
    all_ecans = []
    all_ecans.extend(north_ecans + clyde_ecans)
    all_ecans = df[df['Pay_Number'].isin(all_ecans)]
    ecans_age = np.average(all_ecans['Age'])
    ecans_wte = round(np.sum(all_ecans['WTE']), 1)
    ecans_yos = round(np.average(all_ecans['Years_Of_Service']), 1)

    ecat_yos = round(np.average(df[df['Pay_Number'].isin(clyde_ecats)]['Years_Of_Service']))
    ecat_age = round(np.average(df[df['Pay_Number'].isin(clyde_ecats)]['Age']))
    ecat_wte = round(np.sum(df[df['Pay_Number'].isin(clyde_ecats)]['WTE']))

    #all_econs = df[df['Pay_Number'].isin(clyde_econs.extend(north_econs))]
    #print(len(all_econs))

    south_econs = []
    south_ecans = []
    gerilist = ['G9842943', 'G9843278', 'G9843322', 'G9843323', 'G9843398', 'G9843470', 'G9843472', 'G9843478',
     'G9843484', 'G9843487', 'G9843498', 'G9843499', 'G9843500', 'G9843501', 'G9843502', 'G9843503', 'G9843504',
     'G9843505', 'G9843506', 'G9843507', 'G9843509', 'G9843581', 'G9843583', 'G9843609', 'G9843722', 'G9844457',
     'G9845191', 'G9851744', 'G9852233', 'G9852262', 'G9852263', 'G9852263', 'G9852470', 'G9852781', 'G9852900',
     'G9852936', 'G9852973', 'G9853026', 'G9853063', 'G9853128', 'G9853145', 'G9853177', 'G9853204', 'G9853206',
     'G9853348', 'G9853354', 'G9853415', 'G9853432', 'G9853433', 'G9853437', 'G9853450', 'G9853468', 'G9853550',
     'G9853573', 'G9853579', 'G9853599', 'G9853601', 'G9853647', 'G9853835', 'G9853903', 'G9853921', 'G9853925',
     'G9853932', 'G9853934', 'G9853942', 'G9853961', 'G9853978', 'G9853982', 'G9853989', 'G9854005', 'G9854060',
     'G9854095', 'G9855279', 'G9858997', 'G9859478', 'G9861109', 'G9861125', 'G9861678', 'G9861744', 'G9862217',
     'G9862328', 'G9862436', 'G9862568', 'G9868498', 'G9868631', 'G9868639', 'G9868837', 'G9872262', 'G9875345',
     'G9875590', 'G9876091', 'G9878422', 'G9882493', 'G9885544','G9843483']

    geris = pd.read_csv('W:/MFT/Workforce Profiles/geris.csv')
    geris = geris['PayNums'].tolist()
    geris = df[df['Pay_Number'].isin(geris)]

    cons_geris = geris[geris['Sub_Job_Family'] == 'Consultant']
    spec_geris = geris[geris['Sub_Job_Family'] == 'Other']
    train_geris = geris[geris['Sub_Job_Family'] == 'Training Grades']
    geri_hc = ('Headcount', len(cons_geris), len(spec_geris), len(train_geris))
    geri_wte = ('WTE', np.sum(cons_geris['WTE']), np.sum(spec_geris['WTE']), np.sum(train_geris['WTE']))
    geri_age = ('Age', np.average(cons_geris['Age']),np.average(spec_geris['Age']), np.average(train_geris['Age']))
    geri_yos = ('Years of Service', np.average(cons_geris['Years_Of_Service']), np.average(spec_geris['Years_Of_Service']),
                np.average(train_geris['Years_Of_Service']))
    geri_female = ('% Female', (len(cons_geris[cons_geris['Sex'] == 'F']) / len(cons_geris) * 100),
                   (len(spec_geris[spec_geris['Sex'] == 'F']) / len(spec_geris) * 100),
                    (len(train_geris[train_geris['Sex'] == 'F']) / len(train_geris) * 100))
    geri_thisyear = ('Likely to retire this year', np.sum(cons_geris[cons_geris['ProjRet'] == 'Within a year']['WTE']),
                     np.sum(spec_geris[spec_geris['ProjRet'] == 'Within a year']['WTE']),
                     np.sum(train_geris[train_geris['ProjRet'] == 'Within a year']['WTE']))

    geri_over50 = ('Age 50+', np.sum(cons_geris[cons_geris['Over50'] == 1]['WTE']),
                   np.sum(spec_geris[spec_geris['Over50'] == 1]['WTE']),
                   np.sum(train_geris[train_geris['Over50'] == 1]['WTE']))
    geridata = [geri_hc, geri_wte, geri_age,geri_female, geri_yos, geri_thisyear, geri_over50]
    geriframe = pd.DataFrame(geridata, columns=['Metric','Consultant', 'Other', 'Training Grades']).round(1)

    econs = ('ECON',econs_age, econs_wte, econs_yos)
    ecans = ('ECAN', econs_age, ecans_wte, econs_yos)
    ecats = ('ECAT', ecat_age, ecat_wte, ecat_yos)
    econcancats = pd.DataFrame([econs,ecans,ecats], columns=['','Average Age', 'WTE', 'Years of Service']).round(1)
    print(econcancats)
    # print(econs, '\n', ecans, '\n', ecats)
    # geriframe.to_csv('W:/MFT/geriframe.csv', index=False)
    return geriframe, econcancats



    print(geri_hc, '\n', geri_wte, '\n', geri_age, '\n', geri_yos, '\n', geri_thisyear, '\n', geri_over50)
    # spec_geris = df[df['Sub_Job_Family'] == 'Other']
    # print(len(spec_geris))
    # cons_geris[cons_geris['Post_Descriptor'] == 'Consult   Geriatrics'].to_csv('W:/MFT/cons_gers.csv')
    # print(cons_geris['Post_Descriptor'].value_counts())
    # print(np.mean(cons_geris['Age']))
    # print(np.mean(jun_geris['Age']))
    # print(jun_geris['Sub_Job_Family'].value_counts())
    #TODO GERALDINE MARSH
    #TODO RETIREMENT OF GERIATRICIANS
    nurs_cons_alz = ['G0091626']
    AHP_dementia_consultant = ['G7107986']
    AHP_consultant = ['G295396X']
    totalNew = []
    for i in [north_ecans, north_econs, clyde_ecans, clyde_econs, clyde_ecats, nurs_cons_alz, AHP_dementia_consultant, AHP_consultant]:
        totalNew.extend(i)

    for i in totalNew:
        if (~df['Pay_Number'].str.contains(i).any()):
            print(i + df['Pay_Number']+"in")
        # else:
        #     print(i + "not in")


def jobtrain():
    #This pulls the jobtrain report entitled "All jobs with no. vacancies, no. applicants"
    #TODO turn this into a prompt
    rec_data = askopenfilename(initialdir='//ntserver5/generalDB/WorkforceDB/MFT/Workforce Profiles',
                           filetypes=(("CSV File", "*.csv"), ("All Files", "*.*")),
                           title="Choose the relevant recruitment file."
                           )
    jt1 = pd.read_csv(rec_data)
    jt1['Job Live Date'] = pd.to_datetime(jt1['Job Live Date'], dayfirst=True)
    jt1['Month'] = jt1['Job Live Date'].map(lambda x: x.strftime('%m-%Y'))
    jt1['Department'] = jt1['APPROVAL_Cost_Code'].map(deptlookup)
    jt = jt1[jt1['APPROVAL_Cost_Code'].isin(cost_centres)]
    jt_avg_apps = np.average(jt['Count Distinct Candidate ID Number'])
    jt_totalvacancies = np.sum(jt['Job Number of Vacancies'])
    jt['Vacancy WTE'] = jt['APPROVAL_Total_WTE'] * jt['Job Number of Vacancies']
    jt[['Vacancy WTE', 'APPROVAL_Total_WTE', 'Job Number of Vacancies']].to_csv('W:/MFT/z.csv')
    jtlookups = jt[['Job Title for Candidates', 'Location', 'Department', 'Job Status', 'Vacancy WTE']]
    jtloclookup = {row[0]: row[1] for row in jtlookups.values}
    jtdeptlookup = {row[0]: row[2] for row in jtlookups.values}
    jtstatuslookup = {row[0]: row[3] for row in jtlookups.values} #use these if you want department or job status
    jtwtelookup = {row[0]:row[4] for row in jtlookups.values}
    #print(jt['Job Live Date'].iloc[20])

    jt2 = jt[jt['Job Live Date'] < pd.to_datetime('10/01/2019', format='%m/%d/%Y')]
    jt2 = jt[(jt['Job Status'] == 'Complete') | (jt['Job Status'] == 'Conditional')]

    jt3 = jt[jt['Count Distinct Candidate ID Number'] <= 5]

    plt.figure(10)
    plt.style.use('ggplot')
    jtpiv1 = pd.pivot_table(jt, values='Job Number of Vacancies', index='Month', aggfunc=np.sum)
    #jtpiv1.to_csv('W:/MFT/Workforce Profiles/z.csv')
    #jt.groupby('Month')['Job Number of Vacancies']\
    ax = jtpiv1.plot(kind='bar', color='#003087', legend=False)
    jtmax = max(jtpiv1['Job Number of Vacancies'])
    for i, each in enumerate(jtpiv1.index):
        for col in jtpiv1.columns:
            y = round(jtpiv1.ix[each][col], 1)
            ax.text(i+0.05, y+jtmax/20, y)
    plt.title('Vacancies posted by calendar month')
    plt.xlabel('Number of vacancies')
    plt.ylabel('')
    plt.grid(color='#d7dede')
    plt.tight_layout()
    plt.savefig('W:/MFT/Workforce Profiles/jobvacs.png', dpi=300)
    plt.close()
    numpiv = pd.pivot_table(jt2, values='Job Number of Vacancies', index='Job Title for Candidates', aggfunc = np.sum)
    print(numpiv)

    jobpiv = pd.pivot_table(jt2, values='Count Distinct Candidate ID Number', index='Job Title for Candidates',
                            aggfunc=np.sum)
    jobpiv['Job Title'] = jobpiv.index
    jobpiv['Location'] = jobpiv['Job Title'].map(jtloclookup)
    #jobpiv['Department'] = jobpiv['Job Title'].map(jtdeptlookup)
    #jobpiv['Status'] = jobpiv['Job Title'].map(jtstatuslookup)
    jobpiv = jobpiv.sort_values('Count Distinct Candidate ID Number').head(15)
    jobpiv = jobpiv.rename(columns={'Count Distinct Candidate ID Number':'Applicants'})
    jobpiv = jobpiv[['Job Title', 'Location','Applicants']]
    jobpiv['Job Title'] = jobpiv['Job Title'].str.replace('&amp;', '&')

    #TODO livedate
    #TODO pivot of least popular jobs (subject to livedate) - add job title, location & department
    return jt_avg_apps, jt_totalvacancies, jobpiv, len(jt3)

def basestats(x, title):
    headcount = (len(x))
    depts_count = len(x['department'].unique())
    wte = round(sum(x['WTE']), 1)
    female = round(len(x[x['Sex'] == 'F']) / headcount * 100, 1)
    male = round(100 - female, 1)
    age = round(sum(x['Age']) / headcount, 1)
    unregistered = len(x[x['Pay_Band'].isin(['2', '3', '4'])])
    registered = headcount - unregistered
    print("Registered: " + str(registered), "Unregistered: " + str(unregistered))

    head_table = {'Headcount': str(headcount), 'WTE':str(wte), 'Female %': str(female)+"%", 'Average age':str(age),
                  'Number of Depts':str(depts_count)}
    table1 = pd.DataFrame(list(head_table.items()), columns = ['Metric', 'Quantity'])

    bands = {'2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, 'A': 0, 'B': 0, 'C': 0, 'D': 0, '9':0,
             'Medical and Dental': 0}
    for j in x['Pay_Band'].unique():
        y = len(x[x['Pay_Band'] == j])
        bands[j] = y
    bands['M&D'] = bands.pop('Medical and Dental')



def agecounts(i):
    plt.figure(0)
    plt.style.use('seaborn')
    agecounts = pd.value_counts(df['Age'].values).sort_index()

    graph = df.plot(kind='bar', x='Age', y='WTE')
    plt.legend('')
    graph = agecounts.plot.bar(color = '#003087')
    plt.title(i+' - Age Demography')
    plt.ylabel('Headcount')
    plt.xlabel('Age')

    for label in graph.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)


    plt.savefig('W:/MFT/Workforce Profiles/ageprofile.png', dpi=300)
    plt.close()

    age_group_counts = pd.value_counts(df['Age Group'].values).sort_index()
    graph = age_group_counts.plot.bar(color='#003087', hatch='/')

    plt.title(i+' - Age Demography')
    plt.grid(color = '#d7dede')
    plt.ylabel('Headcount')
    plt.xlabel('Age Group')
    plt.tight_layout()
    plt.savefig('W:/MFT/Workforce Profiles/ageprofile.png', dpi=300)
    plt.close()



def subDir1WTE():
    plt.style.use('ggplot')
    subdirpiv = pd.pivot_table(df, values='WTE', index='Sub-Directorate 1', aggfunc = np.sum)
    ax = subdirpiv.plot(kind = 'barh', color='#003087', legend=False)
    subdirmax = max(subdirpiv['WTE'])

    for i, each in enumerate(subdirpiv.index):
        for col in subdirpiv.columns:
            x = round(subdirpiv.ix[each][col], 1)
            ax.text(x+ (subdirmax/10), i, x)

    #graph2 = df.groupby('Sub-Directorate 1')['WTE'].sum().plot(kind = 'barh', color = '#003087')
    #graph2 = subdir.plot.barh()

    plt.title('WTE by Sub-Directorate')
    plt.xlabel('WTE')
    plt.grid(color='#d7dede')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('W:/MFT/Workforce Profiles/subdir1.png', dpi=300)
    plt.close()

def jobFamWTE():

    plt.style.use('seaborn')
    jobfampiv = pd.pivot_table(df, values='WTE', index='Job_Family', aggfunc=np.sum)
    ax = jobfampiv.plot(kind='barh', color='#003087', legend=False)
    #graph3 = df.groupby('Job_Family')['WTE'].sum().plot(kind = 'barh', color = '#003087')
    jobfammax = max(jobfampiv['WTE'])
    print(jobfammax/30)

    for i, each in enumerate(jobfampiv.index):
        for col in jobfampiv.columns:
            x = round(jobfampiv.ix[each][col], 1)
            ax.text(x + jobfammax/20, i, x)
    plt.ylabel('Job Family')
    plt.xlabel('WTE')
    plt.grid(color='#d7dede')
    plt.title('WTE by Job Family')
    plt.tight_layout()
    plt.savefig('W:/MFT/Workforce Profiles/jobfam.png', dpi=300)
    plt.close()
#todo exclude training grades
#todo differences between partnership/acute for a doctor agewise
#todo composition of job family in partnerships/acute



def payBandWTE():
    plt.style.use('seaborn')


    #capture medical and dental subdir1
    df['PayBand3'] = df['PayBand3'].map({'A': '8A', 'B': '8B', 'C': '8C', 'D': '8D', '2': '2',
                                         '3': '3', '4': '4', '5': '5', '6': '6', '7': '7',
                                         '9': '9', 'Consultant': 'Consultant', 'Other': 'Other (Career Grades)',
                                         'Training Grades': 'Training Grades'})
    paybandpiv = pd.pivot_table(df, values='WTE', index='PayBand3', aggfunc = np.sum)
    ax = paybandpiv.plot(kind='bar', color='#003087', legend=False)
    paymax = max(paybandpiv['WTE'])
    for i, each in enumerate(paybandpiv.index):
        for col in paybandpiv.columns:
            y = round(paybandpiv.ix[each][col],1)
            ax.text(i-0.25, y+paymax/20, y)
    plt.ylabel('WTE')
    x_axis = ax.xaxis
    x_axis.label.set_visible(False)
    #plt.legend(False)
    plt.title('WTE by Pay Band')
    plt.grid(color='#d7dede')
    #plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('W:/MFT/Workforce Profiles/paybands.png', dpi=300)
    plt.close()

def retirementProj():


    overallRetirement = pd.pivot_table(df, values='WTE', index='ProjRet', aggfunc=np.sum)

    overallRetirement = overallRetirement[overallRetirement.index != '>10 years']
    plt.style.use('seaborn')
    ax = overallRetirement.plot(kind='bar', color = '#003087', legend=False)
    retmax = max(overallRetirement['WTE'])
    for i, each in enumerate(overallRetirement.index):
        for col in overallRetirement.columns:
            y = round(overallRetirement.ix[each][col],1)
            ax.text(i, y+retmax/20, y)

    plt.ylabel('WTE')
    plt.grid(color='#d7dede')
    plt.xlabel('Projected Retirement')
    plt.tight_layout()
    plt.savefig('W:/MFT/Workforce Profiles/retirement.png', dpi = 300)
    plt.close()


def absenceTypes():


    print(abs.columns)
    abspiv = pd.pivot_table(abs, values='Absence WTE', index='Department', aggfunc=np.sum)
    abspiv['Department'] = abspiv.index
    abspiv = abspiv.rename(columns={'Absence WTE':'Sickness WTE'})
    abspiv['Inpost WTE'] = abspiv['Department'].map(wtelookup_dept)
    abspiv['A/L WTE'] = abspiv['Department'].map(annWTE_dept)
    abspiv['Maternity WTE'] = abspiv['Department'].map(matWTE_dept)
    abspiv['Total Absence'] = abspiv['Department'].map(totalAbs_dept)
    abspiv['Other WTE'] = abspiv['Total Absence'] - abspiv['A/L WTE'] - abspiv['Sickness WTE'] - \
                              abspiv['Maternity WTE']
    abspiv['Overall Absence %'] = abspiv['Total Absence'] / abspiv['Inpost WTE'] * 100
    abspiv = abspiv[abspiv['Inpost WTE'] >= 20]
    abspiv = abspiv[['Department', 'Inpost WTE', 'Sickness WTE', 'A/L WTE', 'Maternity WTE', 'Other WTE',
                     'Overall Absence %']]
    abspiv = abspiv.sort_values('Overall Absence %', ascending=False).head(15).round(1)
    print(abs.columns)
    print(np.sum(abs['WTE']))
    sick_abs = np.sum(abs['Absence WTE'])
    annual_abs = np.sum(abs['Annual WTE'])
    mat_abs = np.sum(abs['Maternity WTE'])
    other_abs = np.sum(abs['Total Abs WTE']) - sick_abs - mat_abs - annual_abs
    serv_deliv = np.sum(abs['WTE']) - np.sum(abs['Total Abs WTE'])
    abs_pie_labels = ['Service Delivery','Sickness Absence', 'Annual Leave', 'Maternity Leave', 'Other Absence']
    abs_pie_sizes = [serv_deliv,sick_abs, annual_abs, mat_abs, other_abs]
    explode2 = (0, 0.1, 0, 0)
    explode = (0, 0.1, 0.1, 0.1, 0.1)
    fig1, ax1 = plt.subplots()
    pie_wedge = ax1.pie(abs_pie_sizes, explode=explode, labels=abs_pie_labels, autopct='%1.1f%%',
            shadow=False, startangle=180)

    pie_wedge[0][0].set_facecolor('white')
    pie_wedge[0][0].set_edgecolor('#003087')
    ax1.axis('equal')
    plt.title("WTE Allocation by Type")
    plt.savefig('W:/MFT/Workforce Profiles/pie.png', dpi=300)
    plt.close()

    return abspiv

def sortoutbank():
    # path = 'W:/Bank Data v2/'
    # grades = eval(open(path + 'Grades_dic.txt').read())
    # agencies = eval(open(path + 'Agency_dic.txt').read())
    bank = bank_data
    #bank = pd.read_excel('W:/MFT/Workforce Profiles/bankdata-Sept.xlsm')

    bank = bank[bank['Cost Centre'].isin(cost_centres)]
    print(bank.head(10))

    renames = {'Nursing': 'Nursing & Midwifery',
               'Admin & Clerical': 'Administrative Services',
               'Midwifery': 'Nursing & Midwifery'}
    bank['Staff Group'] = bank['Staff Group'].map(renames)
    bank['Requested'] = 1
    bank['Filled-Bank'] = np.where((bank['Staff'].notnull()) & (bank['Agency'].isnull()), 1, '')
    bank['Filled-Agency'] = np.where((bank['Staff'].notnull()) & (bank['Agency'].notnull()), 1, '')

    bank['Agency Name'] = np.where(bank['Filled-Agency'] == '1', bank['Agency'].map(agencies),
                                   np.where(bank['Filled-Bank'] == '1', 'Bank', 'Unfilled'))
    bank.loc[bank['Actual Hours'] == '24:00', 'Actual Hours'] = 0  # to fix 24 or 23:30hrs shifts that are actually 0
    bank.loc[bank['Actual Hours'] == '23:30', 'Actual Hours'] = 0

    bank['Actual Hours'] = bank['Actual Hours'].replace(0, None)
    bank['Actual Hours'] = bank['Actual Hours'].fillna(0)
    bank['Actual Hours'] = bank['Actual Hours'].apply(lambda x: 0 if (type(x) == datetime.datetime) else x)
    bank['Hours Requested'] = bank['Actual Hours'].astype(str)

    # convert actual hours to decimal
    bank['Hours Requested'] = bank['Hours Requested'].str.split(':').apply(
        lambda x: 0 if x == ['0'] else int(x[0]) + int(x[1]) / 60)

    bankfill = bank[bank['Filled-Bank'] == '1']
    agencyfill = bank[bank['Filled-Agency'] == '1']
    bankpiv_dept = pd.pivot_table(bank.round({'Hours Requested':1}), values='Hours Requested', index='Department',
                                  aggfunc = np.sum)
    bankpiv_dept['Department'] = bankpiv_dept.index
    bankpiv_dept = bankpiv_dept[['Department','Hours Requested']]
    bankpiv_dept = bankpiv_dept.sort_values('Hours Requested', ascending=False).head(15).round(1)
    bankpiv_dept['WTE Requested'] = round(bankpiv_dept['Hours Requested'] / (4.5*37.5), 1)
    bankpiv_dept['Overall Absence WTE'] = bankpiv_dept['Department'].map(totalAbs_dept)
    bankpiv_dept['Net Bank Requests'] = bankpiv_dept['WTE Requested'] - bankpiv_dept['Overall Absence WTE']
    bankpiv_dept = bankpiv_dept.round(1)
    bankpiv_sub = pd.pivot_table(bank.round({'Hours Requested':1}), values='Hours Requested', index='Sub-Directorate 1',
                                 aggfunc = np.sum)

    bankpiv_sub['Sub-Directorate 1'] = bankpiv_sub.index
    bankpiv_sub = bankpiv_sub[['Sub-Directorate 1', 'Hours Requested']]

    bankpiv_sub = bankpiv_sub.sort_values('Hours Requested', ascending=False).round(1)

    bankpiv_sub['WTE Requested'] = round(bankpiv_sub['Hours Requested'] / (4.5 * 37.5), 1)
    bankpiv_sub['Overall Absence WTE'] = bankpiv_sub['Sub-Directorate 1'].map(totalAbs_sd)
    bankpiv_sub['Net Bank Requests'] = bankpiv_sub['WTE Requested'] - bankpiv_sub['Overall Absence WTE']
    bankpiv_sub = bankpiv_sub.round(1)

    print(bankpiv_sub)
    total_reqs = len(bank)
    total_hours = np.sum(bank['Hours Requested'])



    return bankfill, agencyfill, bankpiv_dept, bankpiv_sub, total_reqs, total_hours


def average_x(jobfam, flag):
    avx = df[df['Job_Family'] == jobfam]
    if len(avx) == 0:
        return False
    if flag == 1:
        avx = avx[avx['Reg/Unreg'] == 'Registered']
        numstaff = len(avx)
        numleavers = len(leavers[(leavers['Job Family'] == jobfam) & (leavers['Reg/Unreg'] == "Registered")])
        # print(str(numstaff) + " Registered Staff")
        # print(str(numleavers) + " registered leavers in 2018/19")

        xturnover = numleavers / numstaff * 100
        # print(xturnover)
    if flag == 0:
        avx = avx[avx['Reg/Unreg'] == 'Unregistered']
        numstaff = len(avx)
        numleavers = len(leavers[(leavers['Job Family'] == jobfam) & (leavers['Reg/Unreg'] == "Unregistered")])
        # print(str(numstaff) + " Unregistered Staff")
        # print(str(numleavers) + " unreg leavers in 2018/19")

        xturnover = numleavers / numstaff * 100
        # print(xturnover)
    if flag == 2:
        xturnover = 0

    age = int(np.average(avx['Age']))
    #turnover = len(leavers[leavers['Job Family'] == jobfam])/ len(avx) * 100
    #print(turnover)
    print(int(age))
    female = round(len(avx[avx['Sex'] == 'F']) / len(avx) * 100, 1)
    yos = round(np.average(avx['Years_Of_Service']), 1)
    print(yos)
    wte = round(np.sum(avx['WTE']), 1)
    wte_55 = round(np.sum(avx[avx['Age'] > 55]['WTE']) / wte * 100, 1)
    if jobfam == 'Medical and Dental':
        cons = round(np.sum(avx[avx['Sub_Job_Family'] == 'Consultant']['WTE']), 1)
        spec = round(np.sum(avx[avx['Sub_Job_Family'] == 'Other']['WTE']), 1)
        train = round(np.sum(avx[avx['Sub_Job_Family'] == 'Training Grades']['WTE']), 1)
        return age, female, yos, wte, wte_55, cons, spec, train
    print(avx['BandNumeric'].mode())
    #time.sleep(1)
    band = (int(avx['BandNumeric'].median())) #conv to int to fix

    print(band)
    scalepoint = int(np.median(avx['Scale_Point'])) #conv back to int to fix
    print(scalepoint)


    return age, female, yos, band, scalepoint, wte, wte_55, xturnover
def retirement_vuln_depts():
    df['Over55'] = df['Age'].apply(lambda x: 'Staff over 55' if x >= 55 else 'Under 55')
    vulndepts = pd.pivot_table(df, values='WTE', index='department', columns='Over55',
                             aggfunc=np.sum, fill_value=0)
    lookupret_reg = df[(df['Reg/Unreg'] == 'Registered') & (df['Over55'] == 'Staff over 55')]
    regover55 = pd.pivot_table(lookupret_reg, values='WTE', index='department', aggfunc=np.sum).round(1)
    regover55['Department'] = regover55.index
    regretlookups = regover55[['Department', 'WTE']]
    reglookup = {row[0]: row[1] for row in regretlookups.values}
    vulndepts['Total WTE'] = vulndepts['Under 55'] +  vulndepts['Staff over 55']
    vulndepts['% Total Staff over 55'] = vulndepts['Staff over 55'] / vulndepts['Total WTE'] * 100
    vulndepts = vulndepts[vulndepts['Total WTE'] > 20]
    vulndepts = vulndepts.sort_values('% Total Staff over 55', ascending=False).head(10).round(1)
    vulndepts['department'] = vulndepts.index
    vulndepts['Registered Staff over 55'] = vulndepts['department'].map(reglookup)
    vulndepts = vulndepts.rename(columns = {'Registered Staff over 55':"Registered >55", 'Staff over 55':'Total >55',
                                'Total WTE':'Inpost WTE', '% Total Staff over 55':'% Total >55', 'department':'Department'})
    vulndepts = vulndepts[['Department','Inpost WTE','Registered >55','Total >55', '% Total >55']]

    within_year = pd.pivot_table(df[df['ProjRet'] == 'Within a year'], values='WTE', index='department', aggfunc=np.sum)
    within_year['Department'] = within_year.index
    within_year['Total WTE'] = within_year['Department'].map(wtelookup_dept)
    within_year = within_year[within_year['Total WTE'] >= 10]
    within_year['% Retiring Within a Year'] = within_year['WTE'] / within_year['Total WTE'] * 100
    within_year = within_year.sort_values('% Retiring Within a Year', ascending=False)
    within_year = within_year[['Department', 'Total WTE', '% Retiring Within a Year']].round(1).head(15)


    return vulndepts, within_year

def bank_reasons():
    bankr = bank_data
    bankr = bankr[bankr['Cost Centre'].isin(cost_centres)]
    print(bankr.columns)
    bank_reason_piv = pd.pivot_table(bankr, values='Request Id', index='Request Reason', aggfunc=pd.Series.nunique)
    bank_reason_piv['Request Reason'] = bank_reason_piv.index
    #bank_reason_piv = bank_reason_piv.reset_index()
    bank_reason_piv['Number of Requests '] = bank_reason_piv['Request Id']

    sum_reasons =np.sum(bank_reason_piv['Request Id'])
    bank_reason_piv['% Requests'] = round(bank_reason_piv['Number of Requests '] / sum_reasons * 100, 1)
    bank_reason_piv = bank_reason_piv[['Request Reason','Number of Requests ', '% Requests']]
    print(bank_reason_piv)
    return bank_reason_piv
    #bank_reason = bank_reason[bank_reason['Request Type'] > ]
    # bankr = bankr[bankr['']]



def pdfbuilder(i):
    #gframe, eccc = econcans()
    absenceTypes()
    retirementProj()

    agecounts(i)
    subDir1WTE()
    jobFamWTE()
    payBandWTE()
    jtapps, jtvacs, jobtrain_pivot, lessthan5 = jobtrain()
    bank_fill, bank_agency, bank_dept, bank_sub, bank_total_requests, bank_total_hours_requested = sortoutbank()
    abs_pivot = absenceTypes()
    vulndepts, within_year = retirement_vuln_depts()
    bankreasons = bank_reasons()

    doc = SimpleDocTemplate(r"w://MFT/Workforce Profiles/" + i + "-email.pdf", rightMargin=30, leftMargin=30,
                            topMargin=10, bottomMargin=10)

    Story = []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Justify", alignment=TA_CENTER, fontSize=10))
    styles.add(ParagraphStyle(name="avgtab", alignment=TA_CENTER, fontsize=11))
    styles.add(ParagraphStyle(name='tabhead', alignment = TA_CENTER, fontsize=13, textColor=colors.HexColor("#E8EDEE")))
    styles.add(ParagraphStyle(name="subtitle", alignment = TA_CENTER, fontSize=14, textColor=colors.HexColor("#003087")))
    styles.add(ParagraphStyle(name="subtitle2", alignment=TA_CENTER, fontSize=12, textColor=colors.HexColor("#003087")))
    styles.add(ParagraphStyle(name="subhead", alignment = TA_CENTER, fontSize=24, textColor =colors.HexColor("#005EB8")))
    nhslogo = r'W:/Python/Danny/nhsggclogo.png'
    ageprofimg = Image('W:/MFT/Workforce Profiles/ageprofile.png', 5.5 * inch, 3.7 * inch)
    logo = Image(nhslogo, 0.9 * inch, 0.9 * inch)
    logo.hAlign = 'LEFT'
    header = 'Workforce Profile'
    subhead = Paragraph('Area : '+ i, styles['subhead'])
    subhead2 = Paragraph('Data as of 13th January 2020', styles['subhead'])
    retirement = Image('W:/MFT/Workforce Profiles/retirement.png', 4 * inch, 2.7 * inch)
    subdir1 = Image('W:/MFT/Workforce Profiles/subdir1.png', 5 * inch, 3.7 * inch)
    jobfam = Image('W:/MFT/Workforce Profiles/jobfam.png', 5.5 * inch, 3.7 * inch)
    paybands = Image('W:/MFT/Workforce Profiles/paybands.png', 5.5 * inch, 3.7 * inch)
    placeholder = Image('W:/MFT/Workforce Profiles/placeholder.png', 3 * inch, 3 * inch)
    jobvacs = Image('W:/MFT/Workforce Profiles/jobvacs.png', 5.5 * inch, 3.7 * inch)
    pie = Image('W:/MFT/Workforce Profiles/pie.png', 5.5 * inch, 3.7 * inch)
    bandav = np.nanmedian(df['BandNumeric'])
    headcount = (len(df))
    avage = str(round(np.average(df['Age']), 1))
    totalwte = str(round(sum(df['WTE']), 1))
    female = str(round(len(df[df['Sex'] == 'F']) / headcount * 100, 1))
    retiring_year = sum(df[df['ProjRet'] == 'Within a year']['WTE'])
    retiring_percent = retiring_year / sum(df['WTE']) * 100

    depts_count = str(len(df['department'].unique()))






    headertable = Table([[logo, header]])
    headertable.setStyle(TableStyle([('VALIGN', (1, 0), (1, 0), 'TOP'),
                                     ('FONTSIZE', (1,0), (1,0), 48),
                                     ('TEXTCOLOR', (1,0), (1,0), colors.HexColor('#005EB8'))
                                     ]))


    # deptpiv = pd.pivot_table(df[df['Job_Family'] == 'Nursing And Midwifery'], values='WTE', index='department',
    #                          columns='Reg/Unreg', aggfunc=np.sum, fill_value=0)
    # print(df['Reg/Unreg'].value_counts())
    # print(deptpiv)
    # deptpiv['Total'] = deptpiv['Unregistered'] + deptpiv['Registered']
    # deptpiv = deptpiv.sort_values('Total', ascending=False).head(10).round(1)
    # deptpiv['Department'] = deptpiv.index
    # deptpiv = deptpiv[['Department', 'Registered', 'Unregistered', 'Total']]
    # cctable = Table(np.vstack((list(deptpiv), np.array(deptpiv))).tolist())
    # q=(len(deptpiv.columns)-1, len(deptpiv))
    # cctable.setStyle(TableStyle([('BACKGROUND', (0, 0), (q[0], 0), colors.HexColor("#005EB8")),
    #                              ('TEXTCOLOR', (0, 0), (q[0], 0), colors.HexColor("#E8EDEE")),
    #                               ('FONTSIZE', (0, 1), (q[0], q[1]), 8),
    #                               ('ALIGN', (1, 1), (q[0], q[1]), 'CENTER'),
    #                               ('BOX', (0, 1), (q[0], q[1]), 0.006 * inch, colors.black),
    #                               ('BOX', (0, 0), (q[0], 0), 0.006 * inch, (0, 0, 0))
    #                               ]))

    subdir1piv = pd.pivot_table(df[df['Job_Family'] == 'Nursing and Midwifery'], values='WTE', index='Sub-Directorate 1', columns='Reg/Unreg',
                                aggfunc=np.sum, fill_value=0)
    print(subdir1piv)
    subdir1piv.to_csv('W:/MFT/Workforce Profiles/x.csv')
    subdir1piv['Total'] = subdir1piv['Unregistered'] + subdir1piv['Registered']
    subdir1piv = subdir1piv.sort_values('Total', ascending=False).round(1)
    subdir1piv['Sub-Directorate 1'] = subdir1piv.index
    subdir1piv = subdir1piv[['Sub-Directorate 1', 'Registered', 'Unregistered', 'Total']]
    subdir1piv['% Registered'] = round(subdir1piv['Registered'] / subdir1piv['Total'] * 100, 1)
    subdir1piv['% Unregistered'] = 100 - subdir1piv['% Registered']
    subdir1piv['% Unregistered'] = subdir1piv['Unregistered'].astype(str) + " (" + round(subdir1piv['% Unregistered'], 1).astype(
        str) + "%)"
    subdir1piv['% Registered'] = subdir1piv['Registered'].astype(str) + " (" + subdir1piv['% Registered'].astype(str) + "%)"
    subdir1piv.rename(columns={'Registered':'placholder1', 'Unregistered':'placholder2',
                               '% Registered': 'Registered', '% Unregistered':'Registered'})
    subdir1piv = subdir1piv[['Sub-Directorate 1','% Registered', '% Unregistered', 'Total']]


    sd1table = Table(np.vstack((list(subdir1piv), np.array(subdir1piv))).tolist())
    print(sd1table)
    q = (len(subdir1piv.columns) - 1, len(subdir1piv))
    sd1table.setStyle(TableStyle([('BACKGROUND', (0, 0), (q[0], 0), colors.HexColor("#005EB8")),
                                  ('TEXTCOLOR', (0, 0), (q[0], 0), colors.HexColor("#E8EDEE")),
                                  ('FONTSIZE', (0, 1), (q[0], q[1]), 8),
                                  ('ALIGN', (1, 1), (q[0], q[1]), 'CENTER'),
                                  ('BOX', (0, 1), (q[0], q[1]), 0.006 * inch, colors.black),
                                  ('BOX', (0, 0), (q[0], 0), 0.006 * inch, (0, 0, 0))
                                  ]))


    q = (len(vulndepts.columns) - 1, len(vulndepts))
    vulndeptstable = Table(np.vstack((list(vulndepts), np.array(vulndepts))).tolist())
    vulndeptstable.setStyle(TableStyle([('BACKGROUND', (0, 0), (q[0], 0), colors.HexColor("#005EB8")),
                                   ('TEXTCOLOR', (0, 0), (q[0], 0), colors.HexColor("#E8EDEE")),
                                   ('FONTSIZE', (0, 1), (q[0], q[1]), 8),
                                   ('ALIGN', (1, 1), (q[0], q[1]), 'CENTER'),
                                   ('BOX', (0, 1), (q[0], q[1]), 0.006 * inch, colors.black),
                                   ('BOX', (0, 0), (q[0], 0), 0.006 * inch, (0, 0, 0))
                                   ]))
    q = (len(bank_dept.columns) - 1, len(bank_dept))
    bankdepttable = Table(np.vstack((list(bank_dept), np.array(bank_dept))).tolist())
    bankdepttable.setStyle(TableStyle([('BACKGROUND', (0, 0), (q[0], 0), colors.HexColor("#005EB8")),
                                        ('TEXTCOLOR', (0, 0), (q[0], 0), colors.HexColor("#E8EDEE")),
                                        ('FONTSIZE', (0, 1), (q[0], q[1]), 8),
                                        ('ALIGN', (1, 1), (q[0], q[1]), 'CENTER'),
                                        ('BOX', (0, 1), (q[0], q[1]), 0.006 * inch, colors.black),
                                        ('BOX', (0, 0), (q[0], 0), 0.006 * inch, (0, 0, 0))
                                        ]))
    q = (len(bank_sub.columns) - 1, len(bank_sub))
    banksdtable = Table(np.vstack((list(bank_sub), np.array(bank_sub))).tolist())
    banksdtable.setStyle(TableStyle([('BACKGROUND', (0, 0), (q[0], 0), colors.HexColor("#005EB8")),
                                        ('TEXTCOLOR', (0, 0), (q[0], 0), colors.HexColor("#E8EDEE")),
                                        ('FONTSIZE', (0, 1), (q[0], q[1]), 8),
                                        ('ALIGN', (1, 1), (q[0], q[1]), 'CENTER'),
                                        ('BOX', (0, 1), (q[0], q[1]), 0.006 * inch, colors.black),
                                        ('BOX', (0, 0), (q[0], 0), 0.006 * inch, (0, 0, 0))
                                        ]))
    q = (len(jobtrain_pivot.columns) - 1, len(jobtrain_pivot))
    jttable = Table(np.vstack((list(jobtrain_pivot), np.array(jobtrain_pivot))).tolist())
    jttable.setStyle(TableStyle([('BACKGROUND', (0, 0), (q[0], 0), colors.HexColor("#005EB8")),
                                        ('TEXTCOLOR', (0, 0), (q[0], 0), colors.HexColor("#E8EDEE")),
                                        ('FONTSIZE', (0, 1), (q[0], q[1]), 6),
                                        ('ALIGN', (0, 1), (1, q[1]), 'LEFT'),
                                        ('ALIGN', (2,1,), (q[0], q[1]), 'CENTER'),
                                        ('BOX', (0, 1), (q[0], q[1]), 0.006 * inch, colors.black),
                                        ('BOX', (0, 0), (q[0], q[1]), 0.006 * inch, (0, 0, 0))
                                        ]))

    header = Paragraph('<b>'+str(i)+' Staff</b>', styles['tabhead'])
    nursed = average_x('Nursing and Midwifery', 1)
    nurse = [Paragraph('<b>Registered Nurse</b>', styles['avgtab']),
           Paragraph('WTE: '+str(nursed[5])+' ('+str(nursed[6])+' % aged >55)', styles['avgtab']),
           #Paragraph('Average age: '+str(round(nursed[0])), styles['avgtab']),
           #Paragraph(str(nursed[1])+'% female, '+str(1-nursed[1])+ '% male', styles['avgtab']),
           Paragraph(str(nursed[2])+' Average Years of Service', styles['avgtab']),
           Paragraph('Turnover: ' + str(round(nursed[7], 1)) + '%', styles['avgtab']),
           Paragraph('Most Common Band: '+str(nursed[3]), styles['avgtab']),
           Paragraph('Average Increment Point: '+str(nursed[4]), styles['avgtab'])]

    hcswd = average_x('Nursing and Midwifery', 0)
    hcsw = [Paragraph('<b>Healthcare Support Worker</b>', styles['avgtab']),
           # Paragraph('Average age: '+str(round(hcswd[0])), styles['avgtab']),
           Paragraph('WTE: ' + str(hcswd[5]) + ' (' + str(hcswd[6]) + ' % aged >55)', styles['avgtab']),
           #Paragraph(str(hcswd[1])+'% female', styles['avgtab']),
           Paragraph(str(hcswd[2])+' Average Years of Service', styles['avgtab']),
           Paragraph('Turnover: ' + str(round(hcswd[7], 1)) + '%', styles['avgtab']),
           Paragraph('Most Common Band: '+str(hcswd[3]), styles['avgtab']),
           Paragraph('Average Increment Point: '+str(hcswd[4]), styles['avgtab'])]
    medicd = average_x('Medical and Dental',2)
    medic = [Paragraph('<b>Medical and Dental</b>', styles['avgtab']),
           # Paragraph('Average age: '+str(round(medicd[0])), styles['avgtab']),
           Paragraph('WTE: ' + str(medicd[3]) + ' (' + str(medicd[4]) + ' % aged >55)', styles['avgtab']),
           # Paragraph(str(medicd[1])+'% female', styles['avgtab']),
           Paragraph(str(medicd[2])+' Average Years of Service' , styles['avgtab']),
           Paragraph('Consultant WTE: '+str(medicd[5]), styles['avgtab']),
           Paragraph('Training Grades WTE: ' + str(medicd[7]), styles['avgtab']),
           Paragraph('Other WTE: '+str(medicd[6]), styles['avgtab'])]
    ahpd = average_x('Allied Health Profession', 2)
    if ahpd is False:
        AHP = [Paragraph('<b>Allied Health Profession</b>', styles['avgtab']),
               Paragraph('There are no AHPs in this dataset.', styles['avgtab'])]
    else:
        AHP = [Paragraph('<b>Allied Health Profession</b>', styles['avgtab']),
           # Paragraph('Average age: '+str(round(ahpd[0])), styles['avgtab']),
           Paragraph('WTE: ' + str(ahpd[5]) + ' (' + str(ahpd[6]) + ' % aged >55)', styles['avgtab']),
           #Paragraph(str(ahpd[1])+' % female', styles['avgtab']),
           Paragraph(str(ahpd[2])+' Average Years of Service', styles['avgtab']),
           Paragraph('Turnover: ' + str(round(ahpd[7], 1)) + '%', styles['avgtab']),
           Paragraph('Most Common Band: '+str(ahpd[3]), styles['avgtab']),
           Paragraph('Average Increment Point: '+str(ahpd[4]), styles['avgtab'])]

    avgxtable = Table([[header],[nurse, hcsw], [medic, AHP]], colWidths=[2.5*inch] * 2)
    avgxtable.setStyle(TableStyle([('BACKGROUND', (0,0), (0,0), colors.HexColor("#005EB8")),
                                   ('SPAN', (0,0), (1,0)),
                                   ('ALIGNMENT', (0,0), (1,0), 'CENTER'),
                                   ('VALIGN', (0, 1), (1,2), 'TOP'),
                                   ('BOX', (0,0), (1,2), 0.006 * inch, colors.black),
                                   ('BOX', (0, 1), (0, 2), 0.006 * inch, colors.black),
                                   ('BOX', (1, 1), (1, 2), 0.006 * inch, colors.black),
                                   ('BOX', (0,2), (1,2), 0.006 * inch, colors.black)
                                    ]))
    q = (len(abs_pivot.columns) - 1, len(abs_pivot))
    abs_table1 = Table(np.vstack((list(abs_pivot), np.array(abs_pivot))).tolist())
    abs_table1.setStyle(TableStyle([('BACKGROUND', (0,0), (q[0],0), colors.HexColor("#005EB8")),
                                    ('FONTSIZE', (0, 0), (q[0], q[1]), 8),
                                    ('ALIGN', (1, 1), (q[0],q[1]), 'CENTER'),
                                    ('TEXTCOLOR', (0, 0), (q[0], 0), colors.HexColor("#E8EDEE")),
                                    ('BOX', (0,0), (q[0], 0), 0.006 * inch, colors.black),
                                    ('BOX', (0,0), (q[0], q[1]), 0.006 * inch, colors.black)
                                    ]))
    q = (len(within_year.columns) - 1, len(within_year))
    wityear = Table(np.vstack((list(within_year), np.array(within_year))).tolist(), colWidths=[1*inch] * 3)
    print(wityear)
    print("Test")
    print("q[0] = "+str(q[0]) + '\n' + 'q[1] = ' + str(q[1]))
    wityear.setStyle(TableStyle([('BACKGROUND', (0,0), (q[0],0), colors.HexColor("#005EB8")),
                                    ('FONTSIZE', (0, 0), (q[0], q[1]), 6),
                                    ('ALIGN', (1, 1), (q[0],q[1]), 'CENTER'),
                                    ('ALIGN', (0,1), (0,q[1]), 'LEFT'),
                                    ('ALIGN', (1,0), (q[0],0), 'CENTER'),
                                    ('TEXTCOLOR', (0, 0), (q[0], 0), colors.HexColor("#E8EDEE")),
                                    ('BOX', (0,0), (q[0], 0), 0.006 * inch, colors.black),
                                    ('BOX', (0,0), (q[0], q[1]), 0.006 * inch, colors.black),
                                    ]))
    retirement_table = Table([[retirement, wityear]])
    retirement_table.setStyle((TableStyle([('VALIGN', (0,0), (0,0), 'TOP')])))


    #TODO FIX THIS
    q = (len(bankreasons.columns) - 1, len(bankreasons))
    bankreasons_table = Table(np.vstack((list(bankreasons), np.array(bank_reasons()))).tolist(), colWidths=[1.4 * inch] * 3)
    bankreasons_table.setStyle(TableStyle([('BACKGROUND', (0,0), (q[0],0), colors.HexColor("#005EB8")),
                                    ('FONTSIZE', (0, 0), (q[0], q[1]), 8),
                                    ('ALIGN', (1, 1), (q[0],q[1]), 'CENTER'),
                                    ('TEXTCOLOR', (0, 0), (q[0], 0), colors.HexColor("#E8EDEE")),
                                    ('BOX', (0,0), (q[0], 0), 0.006 * inch, colors.black),
                                    ('BOX', (0,0), (q[0], q[1]), 0.006 * inch, colors.black)
                                    ]))

    #TODO REMOVE FROM HERE TO NEXT TODO AFTER MFT OLDER PEOPLES
    # q = (len(gframe.columns) - 1, len(gframe))
    # gtable = Table(np.vstack((list(gframe), np.array(gframe))).tolist())
    # gtable.setStyle(TableStyle([('BACKGROUND', (0,0), (q[0],0), colors.HexColor("#005EB8")),
    #                             ('TEXTCOLOR', (0, 0), (q[0], 0), colors.HexColor("#E8EDEE")),
    #                             ('BOX', (0, 0), (q[0], 0), 0.006 * inch, colors.black),
    #                             ('BOX', (0, 0), (q[0], q[1]), 0.006 * inch, colors.black),
    #                             ('BOX', (0, 0), (0, q[1]), 0.006 * inch, colors.black),
    #                             ('ALIGN', (1,1), (q[0],q[1]), 'CENTER')
    #                             ]))
    # q = (len(eccc.columns) - 1, len(eccc))
    # etable = Table(np.vstack((list(eccc), np.array(eccc))).tolist())
    # etable.setStyle(TableStyle([('BACKGROUND', (0, 0), (q[0], 0), colors.HexColor("#005EB8")),
    #                             ('TEXTCOLOR', (0, 0), (q[0], 0), colors.HexColor("#E8EDEE")),
    #                             ('BOX', (0, 0), (q[0], 0), 0.006 * inch, colors.black),
    #                             ('BOX', (0, 0), (q[0], q[1]), 0.006 * inch, colors.black),
    #                             ('BOX', (0, 0), (0, q[1]), 0.006 * inch, colors.black),
    #                             ('ALIGN', (1, 1), (q[0], q[1]), 'CENTER')
    #                             ]))
    #TODO REMOVE TO HERE

    # contentstab = [[Paragraph('<b>Workforce In-Post</b>', styles['avgtab']),
    #        Paragraph('Recruitment Activity', styles['avgtab']),
    #        Paragraph('Retirement Projections', styles['avgtab']),
    #        Paragraph('Longer Term Projections', styles['avgtab']),
    #        Paragraph('Absence Statistics', styles['avgtab']),
    #        Paragraph('Backfill Statistics', styles['avgtab']),
    #        Paragraph('Further Data Requests', styles['avgtab'])]]


    Story.append(headertable)
    Story.append(Spacer(1, 60))
    Story.append(subhead)
    Story.append(Spacer(1, 60))
    Story.append(subhead2)
    Story.append(Spacer(1, 136))
    Story.append(Paragraph('<b>Contents</b>', styles['subtitle']))
    Story.append(Spacer(1, 24))
    Story.append(Paragraph('Workforce In-Post', styles['subtitle']))
    Story.append(Spacer(1, 24))
    # Story.append(Paragraph('Key Roles', styles['subtitle']))
    # Story.append(Spacer(1, 24))

    Story.append(Paragraph('Recruitment Activity', styles['subtitle']))
    Story.append(Spacer(1, 24))
    Story.append(Paragraph('Retirement Projections', styles['subtitle']))
    Story.append(Spacer(1, 24))
    Story.append(Paragraph('Longer Term Projections', styles['subtitle']))
    Story.append(Spacer(1, 24))
    Story.append(Paragraph('Absence Statistics', styles['subtitle']))
    Story.append(Spacer(1, 24))
    Story.append(Paragraph('Backfill Statistics', styles['subtitle']))
    Story.append(Spacer(1, 24))
    Story.append(Paragraph('Further Data Requests', styles['subtitle']))
    Story.append(Spacer(1, 24))
    Story.append(PageBreak())
    Story.append(Paragraph('<seqChain order = "level_1 level_2"/> <seq id="level_1"/>.'+
                           'Workforce In-Post (January 2020)', styles['subtitle']))
    Story.append(Spacer(1,12))
    Story.append(Paragraph('<b><seq id = "level_1" inc="no"/>.<seq id="level_2"/></b>. The ' + i + ' workforce comprises of ' + str(f'{headcount:,}') + ' employees (' + str(totalwte) +
        ' WTE) and ' + depts_count + ' departments. The workforce is ' + female + '% female and'
        + ' the average employee is ' + avage + ' years old.', styles['Justify']))

    Story.append(Spacer(1, 12))
    Story.append(ageprofimg)
    Story.append(Spacer(1, 12))
    Story.append(avgxtable)

    Story.append(Spacer(1,12))
    Story.append(subdir1)
    Story.append(Spacer(1, 12))
    Story.append(Paragraph('<b><seq id = "level_1" inc="no"/>.<seq id="level_2"/></b>. The above graph shows the ' +i+ ' workforce, separated by sub-directorate.'
    +'The table below shows the split between registered and unregistered Nursing and Midwifery staff across all '+
                            'sub-directorates involved in the workstream. "Registered" status indicates a pay band of 5'
                            +' or above.', styles['Justify']))
    Story.append(Spacer(1,12))
    Story.append(sd1table)
    Story.append(Spacer(1, 12))
    Story.append(jobfam)
    Story.append(Spacer(1, 12))
    Story.append(Paragraph('<b><seq id = "level_1" inc="no"/>.<seq id="level_2"/></b>. The above graph shows the composition of the '+i+' workforce by Job Family.',
                           styles['Justify']))
    Story.append(Spacer(1, 12))
    Story.append(paybands)
    Story.append(Spacer(1, 12))
    Story.append(Paragraph('<b><seq id = "level_1" inc="no"/>.<seq id="level_2"/></b>. The above graph shows the WTE by Pay Band across the '+i+' workforce.'+
                            'The median banding across this workforce is '+str(int(bandav))+'.', styles['Justify']))
    Story.append(Spacer(1, 24))
    # Story.append(Paragraph('Key Roles', styles['subtitle']))
    # Story.append(Spacer(1,24))
    # Story.append(Paragraph('Geriatricians', styles['subtitle2']))
    # Story.append(Spacer(1, 12))
    # Story.append(Paragraph('The '+i+ ' workforce employed 157 Geriatricians as of September 2019. A demographic '+
    #                        'breakdown is shown in the table below.', styles['Justify']))
    # Story.append(Spacer(1, 24))
    #
    # Story.append(gtable)
    # Story.append(Spacer(1, 24))
    # Story.append(Paragraph('ECONs, ECANs & ECATs', styles['subtitle2']))
    # Story.append(Spacer(1, 12))
    # Story.append(Paragraph('The following table shows a breakdown of demography of ECONs, ECATs & ECANs across '
    #                        +'Clyde and North sectors. This will be amended to include South Sector soon.',
    #                        styles['Justify']))
    # Story.append(Spacer(1, 12))
    #Story.append(etable)
    Story.append(PageBreak())
    Story.append(Paragraph('<seq id = "level_1"/>. Recruitment Activity (June 1st 2019 - January 10th 2020)',
                           styles['subtitle']))

    Story.append(Spacer(1,12))
    Story.append(Paragraph('<b><seq id = "level_1" inc="no"/>.<seq id="level_2"/></b> Between June 1st 2019 and January'
                       +' 10th 2020, cost centres associated with the ' + i +
                       ' workstream created ' + str(jtvacs) +
                       ' new vacancies on JobTrain. These vacancies had an average of ' + str(round(jtapps, 1)) +
                       ' applicants, including ' + str(
        lessthan5) + ' roles with fewer than 5 applicants. The below table shows '
                       + 'the ' + str(len(jobtrain_pivot)) + ' completed vacancies with the fewest applicants.',
                       styles['Justify']))
    Story.append(Spacer(1,12))
    Story.append(jttable)
    Story.append(Spacer(1,12))
    Story.append(jobvacs)
    Story.append(PageBreak())
    Story.append(Paragraph('<seq id = "level_1"/>. Retirement Projections', styles['subtitle']))
    Story.append(Spacer(1,12))
    Story.append(Paragraph('<b><seq id = "level_1" inc="no"/>.<seq id="level_2"/></b>. Across the '+i+
                           ' workforce, an estimated '+str(int(retiring_year))+ ' employees ('+
                               str(round(retiring_percent, 1))+'%) are expected to retire within the next year.'+
                                ' All retirement estimates are calculated from historical (3 year) leaver'
                               +' trends based on job family, sex, pay band and Mental Health Officer status. '
                                +'There are currently ' + str(len(mhos))+
                           ' staff with Mental Health Officer (MHO) status within the '+i+ ' workforce. Traditionally,'
                           +' MHOs can be expected to retire near 55.',
                            styles['Justify']))
    Story.append(Spacer(1, 12))
    Story.append(Paragraph('<b><seq id = "level_1" inc="no"/>.<seq id="level_2"/></b>. It is'+
                               ' important to stress that these are estimates based on the assumption that previous '+
                               "trends will continue. In reality, many factors contribute to an employee's retirement"
                                +' decision and estimates. Employees categorised as "Within a year" also '+
                                 'includes employees who have been estimated to retire earlier than the current '
                                 +'calendar year.',
                            styles['Justify']))
    Story.append(Spacer(1,12))
    Story.append(Paragraph('<b><seq id = "level_1" inc="no"/>.<seq id="level_2"/></b>. The below graph estimates retirement '
                           +'risk for the '+i+' workforce.'+' The below table shows the '+str(len(within_year))+
                               ' departments (WTE >10) with the highest percentage of staff estimated to retire within a year. ',
                               styles['Justify']))

    Story.append(Spacer(1, 12))
    Story.append(retirement_table)
    Story.append(Spacer(1, 12))
    Story.append(Paragraph('<seq id = "level_1"/>. Longer Term - Staff over 55', styles['subtitle']))
    Story.append(Spacer(1,12))
    Story.append(Paragraph('<b><seq id = "level_1" inc="no"/>.<seq id="level_2"/></b>. The table below shows the departments with the highest proportion of staff who are over '
                           +'the age of 55.', styles['Justify']))
    Story.append(Spacer(1, 12))
    Story.append(vulndeptstable)

    Story.append(PageBreak())

    Story.append(Paragraph('<seq id = "level_1"/>. Absence Statistics - December 2019', styles['subtitle']))
    Story.append(Spacer(1,12))
    Story.append(Paragraph('<b><seq id = "level_1" inc="no"/>.<seq id="level_2"/></b>. The below chart shows the proportion of WTE allocated to annual leave, maternity leave'
                           +', sickness and other absence and service delivery.', styles['Justify']))
    Story.append(Spacer(1,12))
    Story.append(pie)
    Story.append(Spacer(1,24))
    Story.append(Paragraph('<b><seq id = "level_1" inc="no"/>.<seq id="level_2"/></b>. The below table shows the '+str(len(abs_pivot))+ ' departments (>20 inpost WTE) with the highest overall absence'
                            +' percentages in the ' +i+ ' workforce in December 2019.', styles['Justify']))
    Story.append(Spacer(1, 12))
    Story.append(abs_table1)
    Story.append(PageBreak())
    Story.append(Paragraph('<seq id = "level_1"/>. Backfill Statistics - December 2019', styles['subtitle']))

    Story.append(Spacer(1, 12))



    Story.append(Paragraph('<b><seq id = "level_1" inc="no"/>.<seq id="level_2"/></b>. The '+i+' workforce made '+ str(f'{round(bank_total_requests, 1):,}') + ' bank fill requests in '+
                         'September 2019, totalling '
                         + str(f'{round(bank_total_hours_requested, 1):,}')+ ' hours. Of these, '+str(f'{len(bank_fill):,}')+
                         ' shifts were filled by bank staff and '+str(len(bank_agency))+ ' were filled by agency staff '
                         +'resulting in a fill rate of '+
                         str(round((len(bank_fill) + len(bank_agency)) / bank_total_requests * 100,1))+'%.',
                         styles['Justify']))
    Story.append(Spacer(1, 12))
    Story.append(Paragraph(
        '<b><seq id = "level_1" inc="no"/>.<seq id="level_2"/></b>. The below table shows the proportion of bank requests '
        + ' for each request type.', styles['Justify']))
    Story.append(Spacer(1, 12))
    Story.append(bankreasons_table)
    Story.append(Spacer(1, 12))

    Story.append(Paragraph('<b><seq id = "level_1" inc="no"/>.<seq id="level_2"/></b>. The table below shows the ' + str(len(bank_dept)) +
                           ' departments with the highest number of bank requests across'
                        + ' the ' + i + ' workforce during September 2019. Net bank requests (i.e. where greater bank '
                        + 'WTE is requested than overall absence WTE) may warrant investigation of workload, capacity, '
                        +'etc.', styles['Justify']))
    Story.append(Spacer(1, 12))
    Story.append(bankdepttable)
    Story.append(PageBreak())
    Story.append(Paragraph('<b><seq id = "level_1" inc="no"/>.<seq id="level_2"/></b>. The table below shows bank usage '
                           +'for all Sub-Directorates within the '+i+' workforce.', styles['Justify']))
    Story.append(Spacer(1, 12))
    Story.append(banksdtable)
    Story.append(Spacer(1,48))
    Story.append(Paragraph('<seq id = "level_1"/>. Further data requests', styles['subtitle']))
    Story.append(Spacer(1, 12))
    Story.append(Paragraph('<b><seq id = "level_1" inc="no"/>.<seq id="level_2"/></b>. If you require more detailed data or further bespoke analysis, please email '
                             + '<u>daniel.furnivall@ggc.scot.nhs.uk</u>', styles['Justify']))
    doc.build(Story)
#
#sortoutbank()
# average_x('Nursing And Midwifery', 0)

# average_x('Nursing And Midwifery', 1)
#absenceTypes()
pdfbuilder("MFT Neurosciences")
#econcans()
#bank_reasons()