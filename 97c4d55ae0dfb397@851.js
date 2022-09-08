function _1(md){return(
md`# End-to-end machine learning project: Telco customer churn`
)}

function _2(md){return(
md`Analyzing IBM telecommunications data (Kaggle dataset)`
)}

function _3(md){return(
md`In order to effectively retain customers, it is crucial that **telecommunication companies** are able to predict **customer churn**. This is why large telecommunications corporations develop models to predict customer behaviour so as to take action accordingly.`
)}

function _4(md){return(
md`In this notebook, we shall build a model that predicts the likelihood of customer churn. To achieve this, customer characteristics will be analysed. These include;
1. demographic information
2. account information
3. services information

Our goal is to obtain a data-driven solution that will allow us to reduce churn rates and as a result, increase customer satisfaction and corporation value. 

`
)}

function _5(md){return(
md`#### Our Data set

The data set contains **nineteen columns (independent variables)** that indicate the **characteristics of the clients** of **SimSim**, a fictional telecommunications corporation. 

The **Churn** column (response variable) indicates whether the customer departed within the last month or not. The category **No** includes the clients that did not leave the company last month, while the category **Yes** contains the clients that decided to terminate their relations with the company. Our **objective** is to **obtain the link between the customer's characteristics and the churn**.
`
)}

function _6(md){return(
md`
#### Project Milestones

The project consists of the following sections:

1. **Data Reading**
2. **Exploratory Data Analysis and Cleaning**
3. **Data Visualization**
4. **Feature Importance**
5. **Feature Engineering**
6. **Setting a baseline**
7. **Splitting the data into training and testing sets**
8. **Assessing multiple algorithms**
9. **Algorithm selected: Gradient Boosting**
10. **Hyperparameter tuning**
11. **Performance of the model**
12. **Drawing conclusions - Summary**`
)}

function _7(md){return(
md`### 1. Data Reading `
)}

function _8(md){return(
md`To start, let's import our data.`
)}

function _telcoData(SimpleData){return(
new SimpleData()
  .loadDataFromUrl({
    url:"https://raw.githubusercontent.com/ealecho/makailabs/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"
  })
)}

function _10(md){return(
md`Let's get a description of our data and also render our data in a table to get feels of the variables we shall be interacting with.`
)}

function _descricbeTelcoData(telcoData){return(
telcoData.clone().describe()
)}

function _12(showTable,descricbeTelcoData){return(
showTable(descricbeTelcoData)
)}

function _13(showTable,telcoData){return(
showTable(telcoData)
)}

function _14(md){return(
md`As shown above, the data set contains **19 independent variables**, i.e. a variable whose variation does not depend on that of another. These can be classified into three groups.

(a). **Demographic Information**
   - \`\`\`gender\`\`\`: Whether the client is female or male (\`\`\`Female\`\`\`, \`\`\`Male\`\`\`).
   - \`\`\`SeniorCitzen\`\`\`: Whether the client is a senior citizen or not (\`\`\`0\`\`\`, \`\`\`1\`\`\`).
   - \`\`\`Partner\`\`\`: Whether the client has a partner or not (\`\`\`Yes\`\`\`, \`\`\`No\`\`\`).
   - \`\`\`Dependants\`\`\`: Whether the client has dependents or not (\`\`\`Yes\`\`\`, \`\`\`No\`\`\`).

(b). **Customer Account Information**
   - \`\`\`tenure\`\`\`: Number of months the customer has stayed with the company(Multiple different numeric Values).
   - \`\`\`Contract\`\`\`: Indicates the customer's current contract type (\`\`\`Month-To-Month\`\`\`, \`\`\`One year\`\`\`, \`\`\`Two year\`\`\` ).
   - \`\`\`PaperlessBilling\`\`\`: Whether the client has a partner or not (\`\`\`Yes\`\`\`, \`\`\`No\`\`\`).
   - \`\`\`PaymentMethod\`\`\`: The customer's payment method (\`\`\`Electronic check\`\`\`, \`\`\`Mailed check\`\`\`, \`\`\`Bank transfer (automatic)\`\`\`, \`\`\`Credit card (automatic)\`\`\`).
   - \`\`\`MonthlyCharges\`\`\`: The amount charged to the customer monthly (Multiple different numeric values).
   - \`\`\`TotalCharges\`\`\`: The total amount charged to the customer(Multiple different numeric values).

(c). **Services Information**
   - \`\`\`PhoneService\`\`\`: Whether the client has a phone service or not (\`\`\`Yes\`\`\`, \`\`\`No\`\`\`).
   - \`\`\`MultipleLines\`\`\`: Whether the client has a partner or not (\`\`\`No phone service\`\`\`,\`\`\`Yes\`\`\`, \`\`\`No\`\`\`).
   - \`\`\`InternetServices\`\`\`: Whether the client is subscribed to Internet Service with the company (\`\`\`DSL\`\`\`, \`\`\`Fiber optic\`\`\`, \`\`\`No\`\`\`).
   - \`\`\`OnlineSecurity\`\`\`: Whether the client has online security or not(\`\`\`No internet service\`\`\`,\`\`\`No\`\`\`, \`\`\`Yes\`\`\`).
   - \`\`\`OnlineBackup\`\`\`: Whether the client has online backup or not(\`\`\`No internet service\`\`\`,\`\`\`No\`\`\`, \`\`\`Yes\`\`\`).
   - \`\`\`DeviceProtection\`\`\`: Whether the client has device protection or not(\`\`\`No internet service\`\`\`,\`\`\`No\`\`\`, \`\`\`Yes\`\`\`).
   - \`\`\`TechSupport\`\`\`: Whether the client has tech support or not(\`\`\`No internet service\`\`\`,\`\`\`No\`\`\`, \`\`\`Yes\`\`\`).
   - \`\`\`StreamingTV\`\`\`: Whether the client has streaming TV or not(\`\`\`No internet service\`\`\`,\`\`\`No\`\`\`, \`\`\`Yes\`\`\`).
   - \`\`\`StreamingMovies\`\`\`: Whether the client has streaming movies or not(\`\`\`No internet service\`\`\`,\`\`\`No\`\`\`, \`\`\`Yes\`\`\`).
   `
)}

function _15(md){return(
md`### 2. Exploratory Data Analysis and Data Cleaning

Here we shall analyse the main characteristics of the dataset using **visualisation methods** and **summary statistics**.  Our objective is to;
- understand the data
- discover patterns and anomalies
- check assumptions before performing further evaluations.

### missing values and data types
At the beginning of an EDA, we want to know as much information as possible, this is when the \`\`\`simpleData.checkValues()\`\`\` method comes in handy. This method tells you how many and what percentage ( count | percentage ) of the values are of a specific type.
`
)}

function _checkTelcoData(telcoData){return(
telcoData.clone().checkValues()
)}

function _17(showTable,checkTelcoData){return(
showTable(checkTelcoData)
)}

function _18(md){return(
md`As shown above, the data set contains **7043 observations** and **21 columns.** Apparently, there are no \`\`\`null\`\`\` values on the data set; however, we observe that \`\`\`seniorCitizen\`\`\`, \`\`\`tenure\`\`\`, \`\`\`monthlyCharges\`\`\` and \`\`\`totalCharges\`\`\` have been wrongly detected as having a data type of \`\`\`string\`\`\`.  These are supposed to be numeric variables, so we need to convert these columns into a **numeric** data type for further analysis. To do so, we use the \`\`\`SimpleData.valuesToInteger()\`\`\` or \`\`\`SimpleData.valuesToFloat()\`\`\` where necessary. While we are at it, let us convert the column names to [camel case convention](https://en.wikipedia.org/wiki/Camel_case) using \`\`\`SimpleData.formatAllKeys()\`\`\` method.`
)}

function _telcoData1(telcoData){return(
telcoData.clone().formatAllKeys()
)}

function _20(showTable,telcoData1){return(
showTable(telcoData1)
)}

function _21(md){return(
md`So Here we shall convert certain fileds to integers in the fields below:`
)}

function _telcoDataCleanNum(telcoData1){return(
telcoData1
.clone()
.valuesToInteger({key:"seniorCitizen"})
.valuesToInteger({key:"tenure"})
.valuesToFloat({key:"monthlyCharges", skipErrors: true})
.valuesToFloat({key:"totalCharges", skipErrors: true})
)}

function _cleanDataCheck(telcoDataCleanNum){return(
telcoDataCleanNum.clone().checkValues()
)}

function _24(showTable,cleanDataCheck){return(
showTable(cleanDataCheck)
)}

function _25(md){return(
md`In our second data check, observe \`\`\`totalCharges\`\`\` has 11 missing values. Of the 7043 observations, 7032 are numbers, while the remaining 11 are empty strings.`
)}

function _26(showTable,telcoDataCleanNum){return(
showTable(telcoDataCleanNum)
)}

function _27(md){return(
md`These 11 observations in \`\`\`totalCharges\`\`\` with missing values also have a tenure of 0. Of course, I would not expect a customer that has not spent any time on the network to have accrued a total charge; therefore, we remove those observations from the data set. We shall use the \`\`\`simpleData.excludeMissingValues()\`\`\` method to remove items (or rows in a spreadsheet way) in which one or more values are missing.`
)}

function _telcoData2(telcoDataCleanNum){return(
telcoDataCleanNum
  .clone()
  .excludeMissingValues({missingValues:[" "]})
)}

function _telcoData2Check(telcoData2){return(
telcoData2
  .clone()
  .checkValues()
)}

function _30(showTable,telcoData2Check){return(
showTable(telcoData2Check)
)}

function _31(showTable,telcoData2){return(
showTable(telcoData2)
)}

function _32(md){return(
md`### Remove customerId column

The \`\`\`customerID\`\`\` column is useless to explain whether or not the customer will churn. Therefore, let's drop it from the data set using the \`\`\`simpleData.removeKey()\`\`\` method.`
)}

function _telcoData3(telcoData2){return(
telcoData2
  .clone()
  .removeKey({ key: "customerId"})
)}

function _34(showTable,telcoData3){return(
showTable(telcoData3)
)}

function _35(md){return(
md`#### Payment method denominations

\`\`\`\`paymentMethod\`\`\`\` has four unique values. Some payment method denominations contain in parenthesis the word automatic. These are too long to be used as tick labels in further visualizations. Let us remove the clarification in parenthesis using the SimpleData.replaceValues() method.`
)}

function _telcoData4(telcoData3){return(
telcoData3
  .clone()
  .replaceValues({
    key: "paymentMethod",
    oldValue: "automatic",
    newValue: "",
    method: "partialString"
  })
)}

function _paymentMethods(telcoData4){return(
telcoData4
  .clone()
  .getUniqueValues({key: "paymentMethod"})
)}

function _38(md){return(
md`### 3. Data Visualization
Let's do some analysis using visualisation.

#### Response Variable
The following **bar plot** shows how the observations correspond to each class of the **response variable**: \`\`no\`\` and \`\`yes\`\`. This is an imbalanced dataset because both classes are not equally distributed among all observations. **no** is clearly the majority class. This might affect our model later because it might lead to false negatives ie. wrongly indicating that a position does not hold.`
)}

function _39(Plot,telcoData4){return(
Plot.plot({
  y:{
  grid: true,
  label:"proportion of observations",
  },
  facet: {
    data:telcoData4.getData(),
  },
  marks: [
    Plot.barY(
      telcoData4.getData(),
      Plot.groupX({ y: "proportion", fill: "sum"}, { 
        x: "churn", 
      })
    )
  ],
  height: 600,
  marginLeft: 50,
  marginBottom: 50,
  width: 674
})
)}

function _40(md){return(
md`Let us use a **normalised stacked bar**  to analyse the **influence of each independent categorical variable in the outcome.**

A **normalised stacked bar** plot makes each column the same height, so it does not help compare total numbers; however **it is perfect for comparing how the response variable varies across all groups of an independent variable.**

On the other hand, we use **histograms** to evaluate the **influence of each independent numeric variable on the outcome.** We have already seen that the data set is imbalanced; therefore, we need to draw a probability density function of each class \`\`\`density=True\`\`\` to compare both distributions properly.
`
)}

function _41(md){return(
md`### (a) Demographic Information
Let's create a stacked percentage bar chart for each demographic attribute **(\`\`\`\`gender\`\`\`\`,\`\`\`\`seniorCitizen\`\`\`\`, \`\`\`\`partner\`\`\`\`,\`\`\`\`dependents\`\`\`\`)**, showing the percentage of \`\`\`\`churn\`\`\`\` for each category of the attribute.`
)}

function _42(Plot,telcoData4){return(
Plot.plot({
    color: {
    legend: true
  },
  y: {
    grid: true,
    percent: true,
    label:"proportion of observations by gender",
  },
  facet: {
    data: telcoData4.getData(),
    x: "gender"
  },
  marks: [
    Plot.barY(telcoData4.getData(), Plot.groupZ({y: "proportion-facet"}, {fill: "churn"})),
    Plot.ruleY([0, 1])
  ],
  height: 400,
  width:400,
  marginLeft: 50,
  marginBottom: 50,
})
)}

function _43(Plot,telcoData4){return(
Plot.plot({
    color: {
    legend: true
  },
  y: {
    grid: true,
    percent: true,
    label:"proportion of observations by seniorCitizen",
  },
  facet: {
    data: telcoData4.getData(),
    x: "seniorCitizen"
  },
  marks: [
    Plot.barY(telcoData4.getData(), Plot.groupZ({y: "proportion-facet"}, {fill: "churn"})),
    Plot.ruleY([0, 1])
  ],
  height: 400,
  width:400,
  marginLeft: 50,
  marginBottom: 50,
})
)}

function _44(Plot,telcoData4){return(
Plot.plot({
    color: {
    legend: true
  },
  y: {
    grid: true,
    percent: true,
    label:"proportion of observations by Partner",
  },
  facet: {
    data: telcoData4.getData(),
    x: "partner"
  },
  marks: [
    Plot.barY(telcoData4.getData(), Plot.groupZ({y: "proportion-facet"}, {fill: "churn"})),
    Plot.ruleY([0, 1])
  ],
  height: 400,
  width:400,
  marginLeft: 50,
  marginBottom: 50,
})
)}

function _dependentsChart(Plot,telcoData4){return(
Plot.plot({
  color: {
    legend: true
  },
  y: {
    grid: true,
    percent: true,
    label:"proportion of observations by Dependents",
  },
  facet: {
    data: telcoData4.getData(),
    x: "dependents"
  },
  marks: [
    Plot.barY(telcoData4.getData(), Plot.groupZ({y: "proportion-facet"}, {fill: "churn"})),
    Plot.ruleY([0, 1])
  ],
  height: 400,
  width:400,
  marginLeft: 50,
  marginBottom: 50,
})
)}

function _46(md){return(
md`We can see that each bar is a category of the independent variable and is subdivided to show the proportion of each response class (**No** and **Yes**). We can draw the **following conclusions** from an analysis of the **demographic attributes**:

- The churn rate of **senior citizens** is almost double that of **young citizens**.
- We do not expect **gender** to have similar predictive power. A similar percentage of churn is shown when a customer is a man or woman.
- Customers with a **partner** churn less than customers with no partner.

`
)}

function _47(md){return(
md`### (b) Customer Account Information - Categorical Variables

As we did with demographic attributes, we evaluate the percentage of \`\`\`\`churn\`\`\`\` for each category of the customer attributes (\`\`\`\`contract\`\`\`\`, \`\`\`\`paperlessBilling\`\`\`\`, \`\`\`\`paymentMethod\`\`\`\`)`
)}

function _contractsChart(Plot,telcoData4){return(
Plot.plot({
  color: {
    legend: true
  },
  y: {
    grid: true,
    percent: true,
    label:"Proportion of observations by Contract",
  },
  facet: {
    data: telcoData4.getData(),
    x: "contract"
  },
  marks: [
    Plot.barY(telcoData4.getData(), Plot.groupZ({y: "proportion-facet"}, {fill: "churn"})),
    Plot.ruleY([0, 1])
  ],
  height: 400,
  width:400,
  marginLeft: 50,
  marginBottom: 50,
})
)}

function _paperlessBillingChart(Plot,telcoData4){return(
Plot.plot({
  color: {
    legend: true
  },
  y: {
    grid: true,
    percent: true,
    label:"Proportion of observations by PaperlessBilling",
  },
  facet: {
    data: telcoData4.getData(),
    x: "paperlessBilling"
  },
  marks: [
    Plot.barY(telcoData4.getData(), Plot.groupZ({y: "proportion-facet"}, {fill: "churn"})),
    Plot.ruleY([0, 1])
  ],
  height: 400,
  width:400,
  marginLeft: 50,
  marginBottom: 50,
})
)}

function _paymentMethodChart(Plot,telcoData4){return(
Plot.plot({
  color: {
    legend: true
  },
  y: {
    grid: true,
    percent: true,
    label:"Proportion of observations by Payment method",
  },
  facet: {
    data: telcoData4.getData(),
    x: "paymentMethod"
  },
  marks: [
    Plot.barY(telcoData4.getData(), Plot.groupZ({y: "proportion-facet"}, {fill: "churn"})),
    Plot.ruleY([0, 1])
  ],
  height: 400,
  width:400,
  marginLeft: 50,
  marginBottom: 50,
})
)}

function _51(md){return(
md` By analyzing **customer account attributions**, we can draw the **following conclusions.**
 - Customers with **month-to-month contracts** have higher churn rates compared to clients with **yearly contracts**
 - Customers who opted for an **electronic check** as paying method are likely to leave the company.
 - Customers subscribed to **paperless billing** churn more than those who are not subscribed.

`
)}

function _52(md){return(
md`### (b) Customer Account Information - Numerical variables

The following plots show the distribution of \`\`\`\`tenure\`\`\`\`, \`\`\`\`monthlyCharges\`\`\`\`, \`\`\`\`totalCharges\`\`\`\` by \`\`\`\`churn\`\`\`\`.For all numeric attributes, the distributions of both classes (**No** and **Yes**) are different, which suggests that all of the attributes will be useful in determining whether or not a customer churns.
`
)}

function _tenureChart(Plot,telcoData4){return(
Plot.plot({
  color: {
    legend: true
  },
  y: {
    grid: true,
  },
  marks: [
    Plot.barY(telcoData4.getData(), Plot.binX({y: "count"}, {x:"tenure",fill: "churn", thresholds: 10})),
    Plot.ruleY([0, 1])
  ],
  height: 400,
  width:400,
  marginLeft: 50,
  marginBottom: 50,
})
)}

function _monthlyChargesChart(Plot,telcoData4){return(
Plot.plot({
  color: {
    legend: true
  },
  y: {
    grid: true,
  },
  marks: [
    Plot.barY(telcoData4.getData(), Plot.binX({y: "count"}, {x:"monthlyCharges",fill: "churn", thresholds: 10})),
    Plot.ruleY([0, 1])
  ],
  height: 400,
  width:400,
  marginLeft: 50,
  marginBottom: 50,
})
)}

function _totalChargesChart(Plot,telcoData4){return(
Plot.plot({
  color: {
    legend: true
  },
  y: {
    grid: true,
  },
  marks: [
    Plot.barY(telcoData4.getData(), Plot.binX({y: "count"}, {x:"totalCharges",fill: "churn", thresholds: 10})),
    Plot.ruleY([0, 1])
  ],
  height: 400,
  width:400,
  marginLeft: 50,
  marginBottom: 50,
})
)}

function _56(md){return(
md`By analysing **the histograms above** we reach the following conclusions;
- The churn rate tends to be larger when **monthly charges** are high.
- New customers (low **tenure**) are more likely to churn.
- Clients with high **total charges** are less likely to leave the company. `
)}

function _57(md){return(
md`### (c). Services Information

Lastly, let's evaluate the target percentage for each category of services columns with stacked bar plots.`
)}

function _phoneServiceChart(Plot,telcoData4){return(
Plot.plot({
  color: {
    legend: true
  },
  y: {
    grid: true,
    percent: true,
    label:"Proportion of observations by Phone Service",
  },
  facet: {
    data: telcoData4.getData(),
    x: "phoneService"
  },
  marks: [
    Plot.barY(telcoData4.getData(), Plot.groupZ({y: "proportion-facet"}, {fill: "churn"})),
    Plot.ruleY([0, 1])
  ],
  height: 400,
  width:400,
  marginLeft: 50,
  marginBottom: 50,
})
)}

function _multipleLinesChart(Plot,telcoData4){return(
Plot.plot({
  color: {
    legend: true
  },
  y: {
    grid: true,
    percent: true,
    label:"Proportion of observations by Multiple Lines",
  },
  facet: {
    data: telcoData4.getData(),
    x: "multipleLines"
  },
  marks: [
    Plot.barY(telcoData4.getData(), Plot.groupZ({y: "proportion-facet"}, {fill: "churn"})),
    Plot.ruleY([0, 1])
  ],
  height: 400,
  width:400,
  marginLeft: 50,
  marginBottom: 50,
})
)}

function _internetServiceChart(Plot,telcoData4){return(
Plot.plot({
  color: {
    legend: true
  },
  y: {
    grid: true,
    percent: true,
    label:"Proportion of observations by Internet Service",
  },
  facet: {
    data: telcoData4.getData(),
    x: "internetService"
  },
  marks: [
    Plot.barY(telcoData4.getData(), Plot.groupZ({y: "proportion-facet"}, {fill: "churn"})),
    Plot.ruleY([0, 1])
  ],
  height: 400,
  width:400,
  marginLeft: 50,
  marginBottom: 50,
})
)}

function _onlineSecurityChart(Plot,telcoData4){return(
Plot.plot({
  color: {
    legend: true
  },
  y: {
    grid: true,
    percent: true,
    label:"Proportion of observations by Online Security",
  },
  facet: {
    data: telcoData4.getData(),
    x: "onlineSecurity"
  },
  marks: [
    Plot.barY(telcoData4.getData(), Plot.groupZ({y: "proportion-facet"}, {fill: "churn"})),
    Plot.ruleY([0, 1])
  ],
  height: 400,
  width:400,
  marginLeft: 50,
  marginBottom: 50,
})
)}

function _onlineBackupChart(Plot,telcoData4){return(
Plot.plot({
  color: {
    legend: true
  },
  y: {
    grid: true,
    percent: true,
    label:"Proportion of observations by Online Backup",
  },
  facet: {
    data: telcoData4.getData(),
    x: "onlineBackup"
  },
  marks: [
    Plot.barY(telcoData4.getData(), Plot.groupZ({y: "proportion-facet"}, {fill: "churn"})),
    Plot.ruleY([0, 1])
  ],
  height: 400,
  width:400,
  marginLeft: 50,
  marginBottom: 50,
})
)}

function _deviceProtectionChart(Plot,telcoData4){return(
Plot.plot({
  color: {
    legend: true
  },
  y: {
    grid: true,
    percent: true,
    label:"Proportion of observations by Device Protection",
  },
  facet: {
    data: telcoData4.getData(),
    x: "deviceProtection"
  },
  marks: [
    Plot.barY(telcoData4.getData(), Plot.groupZ({y: "proportion-facet"}, {fill: "churn"})),
    Plot.ruleY([0, 1])
  ],
  height: 400,
  width:400,
  marginLeft: 50,
  marginBottom: 50,
})
)}

function _techSupportChart(Plot,telcoData4){return(
Plot.plot({
  color: {
    legend: true
  },
  y: {
    grid: true,
    percent: true,
    label:"Proportion of observations by Tech Support",
  },
  facet: {
    data: telcoData4.getData(),
    x: "techSupport"
  },
  marks: [
    Plot.barY(telcoData4.getData(), Plot.groupZ({y: "proportion-facet"}, {fill: "churn"})),
    Plot.ruleY([0, 1])
  ],
  height: 400,
  width:400,
  marginLeft: 50,
  marginBottom: 50,
})
)}

function _streamingTVChart(Plot,telcoData4){return(
Plot.plot({
  color: {
    legend: true
  },
  y: {
    grid: true,
    percent: true,
    label:"Proportion of observations by Streaming TV",
  },
  facet: {
    data: telcoData4.getData(),
    x: "streamingTv"
  },
  marks: [
    Plot.barY(telcoData4.getData(), Plot.groupZ({y: "proportion-facet"}, {fill: "churn"})),
    Plot.ruleY([0, 1])
  ],
  height: 400,
  width:400,
  marginLeft: 50,
  marginBottom: 50,
})
)}

function _streamingMoviesChart(Plot,telcoData4){return(
Plot.plot({
  color: {
    legend: true
  },
  y: {
    grid: true,
    percent: true,
    label:"Proportion of observations by Streaming Movies",
  },
  facet: {
    data: telcoData4.getData(),
    x: "streamingMovies"
  },
  marks: [
    Plot.barY(telcoData4.getData(), Plot.groupZ({y: "proportion-facet"}, {fill: "churn"})),
    Plot.ruleY([0, 1])
  ],
  height: 400,
  width:400,
  marginLeft: 50,
  marginBottom: 50,
})
)}

function _67(md){return(
md`From evaluating the **services attributes** we can draw the **following conclusions**.
  - We do not expect **phone attributes** (\`\`\`\`phoneService\`\`\`\` and \`\`\`\`multipleLines\`\`\`\`) to have significant predictive power. The churn percentage for all classes in both independent variables is nearly identical.
  - Clients with **online security** churn less than those without it.
  - Customers with no **tech support** tend to churn more often than those with tech support.

The above plots show us the **most relevant attributes for detecting churn**. We expect these attributes to be discriminative in our future models.`
)}

function _68(md){return(
md`## 4. Feature Importance`
)}

function _69(md){return(
md`
**Features** or variables are **individual properties** e.g. **contract**, **paperlessBilling**, **totalCharges** generated from a data set and **used as input to ML models.** Usually, we represent features as numerical columns in data sets, but they can also be strings e.g **paymentMethod** in our example.

**Feature Importance** expresses how much each feature **contributes** to the model prediction. It helps us determine the usefulness of a **specific feature** for a current model and prediction. This is usually represented using a score. The higher the score a value has, the more critical it is. Feature importance **scores** are beneficial in the following ways.

- It is possible to determine the relationship between **independent variables** (features) and **dependent variables** (targets).
- By analysing variable importance scores, we could find irrelevant features and exclude them.
- Reducing the number of not-so-important variables will speed up the model or improve its performance.

**Feature Importance** is commonly used as a **tool** for **ML model interpretability**. From scores, it is possible to explain why an ml model makes particular predictions and how it can manipulate features to change its predictions.

**Note:** When writing this, I could not find a JavaScript library that conveniently lets me compute the **mutual information score**(mutual information measures the dependency between two variables). The way **Scikit-Learn** library does with its **metrics package**. **//TODO**

Mutual information gives us a better understanding of our data and also **identifies features** that are completely **independent of our target**. 

Luckily we can already draw from our data visualisations  **gender**, **phoneService**, and **multipleLines** will not have so much predictive power because the churn percentage for all classes(**No** or **Yes**) in all these independent variables is nearly identical.  **Mutual information** allows us to use correlation criteria on nonlinear relationships, unlike the Pearson correlation that only measures linear dependence between two variables. This method of calculating feature importance **detects not only linear relationships but also nonlinear ones.**

These features (**gender**, **phoneService**, and **multipleLines**) should be removed from the dataset before training as they do not provide useful information for predicting the outcome.
`
)}

function _70(md){return(
md`## 5. Feature Engineering

**Features** must be **extracted** from data **and transformed** into a format suitable for the machine learning model. This is called **Feature Engineering**. Let's transform both the numerical and categorical variables before training our model.
- All **categorical attributes** available in the dataset should be **encoded**(transformed) into **numerical labels**
- **Numerical columns** should be transformed into a **common scale**

This will prevent columns with large values from dominating the learning process. 

### Label Encoding 
Categorical values are replaced with numerical values using **label encoding**, which replaces every **category** with a **numerical label**. Let's transform the following binary variables using label encoding; 
 - gender
 - partner
 - dependents
 - paperlessBilling
 - phoneService
 - churn`
)}

function _71(showTable,telcoData4){return(
showTable(telcoData4)
)}

function _telcoLabelEncod(telcoData4)
{
  let telcoDatar = telcoData4
    .clone()
    .replaceValues({
      key: "gender",
      oldValue:"Female",
      newValue:1,
    })
    .replaceValues({
      key: "gender",
      oldValue:"Male",
      newValue:0,
    })
    .replaceValues({
      key: "partner",
      oldValue:"Yes",
      newValue:1,
    })
   .replaceValues({
      key: "partner",
      oldValue:"No",
      newValue:0,
    })
   .replaceValues({
      key: "dependents",
      oldValue:"Yes",
      newValue:1,
    })
   .replaceValues({
      key: "dependents",
      oldValue:"No",
      newValue:0,
    })
    .replaceValues({
      key: "paperlessBilling",
      oldValue:"Yes",
      newValue:1,
    })
   .replaceValues({
      key: "paperlessBilling",
      oldValue:"No",
      newValue:0,
    })
      .replaceValues({
      key: "phoneService",
      oldValue:"Yes",
      newValue:1,
    })
   .replaceValues({
      key: "phoneService",
      oldValue:"No",
      newValue:0,
    })
    .replaceValues({
      key: "churn",
      oldValue:"Yes",
      newValue:1,
    })
   .replaceValues({
      key: "churn",
      oldValue:"No",
      newValue:0,
    })

  return telcoDatar
}


function _73(showTable,telcoLabelEncod){return(
showTable(telcoLabelEncod)
)}

function _74(md){return(
md`### One-Hot Encoding
This creates a **new binary column for each level of categorical variable.** The new column contains zeros and ones indicating the absence or presence of the category in the data. We shall apply one-hot encoding to the following categorical variables: 
- contract
- paymentMethod
- multipleLines
- internetServices
- onlineSecurity
- onlineBackup
- deviceProtection
- techSupport
- streamingTV
- streamingMovies`
)}

function _telcoDf(dfd,telcoLabelEncod){return(
new dfd.DataFrame(telcoLabelEncod.clone().getData())
)}

function _telcoOneHotEncode(dfd,telcoDf,SimpleData)
{
  let one_hot_encoding_columns = ['multipleLines', 'internetService', 'onlineSecurity','onlineBackup', 'deviceProtection', 'techSupport', 'streamingTv',  'streamingMovies', 'contract', 'paymentMethod']
  let telcoDataTransformed = dfd.getDummies(telcoDf,{columns: one_hot_encoding_columns} )

  let telcoDataTransformedExport = dfd.toJSON(telcoDataTransformed)

  return new SimpleData({
  data: telcoDataTransformedExport,
})
}


function _77(showTable,telcoOneHotEncode){return(
showTable(telcoOneHotEncode)
)}

function _78(md){return(
md`### Normalisation

It is the process of transforming **numeric columns** to a **common scale**. In machine learning, some feature values differ from others by multiple times. Features with higher values will dominate the learning process, however it doesnt meant that those variables are more important to predict the target. **Data normalization** transforms multi-scaled data to the same scale. **Normalization** causes all variables to have a similar influence on the model, improving the stability and performance of the learning algorithm.`
)}

function _normalization(dfd,telcoOneHotEncode,SimpleData)
{
  // min-max normalization (numeric variables)
  //let min_max_columns = ['tenure', 'monthlyCharges', 'totalCharges']
  
  let newTelcoDf = new dfd.DataFrame(telcoOneHotEncode.getData())

  let scaler = new dfd.MinMaxScaler()
  scaler.fit(newTelcoDf)

   let telcoDataNormalised = dfd.toJSON(scaler.transform(newTelcoDf))

  return new SimpleData({
  data: telcoDataNormalised,
})

}


function _80(showTable,normalization){return(
showTable(normalization)
)}

function _81(md){return(
md`### Split the data in training and testing sets

`
)}

function _data(dfd,normalization)
{
  let telcoDataTransformed = new dfd.DataFrame(normalization.clone().getData())

  //select independent variables
  let X = telcoDataTransformed.drop({columns: "churn"})

  // select dependent variables
  let Y = telcoDataTransformed.loc({columns: ["churn"]})
  return [X.tensor, Y.tensor]

}


function _83(md){return(
md`## Building a Deep Neural Network`
)}

function _84(md){return(
md`#### Import necessary libraries
We simple-data-library by *TODO* to read and explore our data`
)}

async function _SimpleData()
{
  const sda = await import("https://cdn.skypack.dev/simple-data-analysis");
  return sda.SimpleData
}


function _86(md){return(
md`Import danfojs`
)}

function _dfd(require,print){return(
require("danfojs@1.0.5/lib/bundle.js").catch(() => {
  window.dfd.Series.prototype.print = window.dfd.DataFrame.prototype.print = function () {
    return print(this);
  };
  return window.dfd;
})
)}

function _88(md){return(
md`#### Define some Helper Functions
One shall help us render our charts and the other to render tables`
)}

function _showChart(html){return(
function showChart(htmlOrSvg) {
  return html`${htmlOrSvg}`;
}
)}

function _showTable(Inputs){return(
function showTable(sd) {
  return Inputs.table(sd.getData());
}
)}

export default function define(runtime, observer) {
  const main = runtime.module();
  main.variable(observer()).define(["md"], _1);
  main.variable(observer()).define(["md"], _2);
  main.variable(observer()).define(["md"], _3);
  main.variable(observer()).define(["md"], _4);
  main.variable(observer()).define(["md"], _5);
  main.variable(observer()).define(["md"], _6);
  main.variable(observer()).define(["md"], _7);
  main.variable(observer()).define(["md"], _8);
  main.variable(observer("telcoData")).define("telcoData", ["SimpleData"], _telcoData);
  main.variable(observer()).define(["md"], _10);
  main.variable(observer("descricbeTelcoData")).define("descricbeTelcoData", ["telcoData"], _descricbeTelcoData);
  main.variable(observer()).define(["showTable","descricbeTelcoData"], _12);
  main.variable(observer()).define(["showTable","telcoData"], _13);
  main.variable(observer()).define(["md"], _14);
  main.variable(observer()).define(["md"], _15);
  main.variable(observer("checkTelcoData")).define("checkTelcoData", ["telcoData"], _checkTelcoData);
  main.variable(observer()).define(["showTable","checkTelcoData"], _17);
  main.variable(observer()).define(["md"], _18);
  main.variable(observer("telcoData1")).define("telcoData1", ["telcoData"], _telcoData1);
  main.variable(observer()).define(["showTable","telcoData1"], _20);
  main.variable(observer()).define(["md"], _21);
  main.variable(observer("telcoDataCleanNum")).define("telcoDataCleanNum", ["telcoData1"], _telcoDataCleanNum);
  main.variable(observer("cleanDataCheck")).define("cleanDataCheck", ["telcoDataCleanNum"], _cleanDataCheck);
  main.variable(observer()).define(["showTable","cleanDataCheck"], _24);
  main.variable(observer()).define(["md"], _25);
  main.variable(observer()).define(["showTable","telcoDataCleanNum"], _26);
  main.variable(observer()).define(["md"], _27);
  main.variable(observer("telcoData2")).define("telcoData2", ["telcoDataCleanNum"], _telcoData2);
  main.variable(observer("telcoData2Check")).define("telcoData2Check", ["telcoData2"], _telcoData2Check);
  main.variable(observer()).define(["showTable","telcoData2Check"], _30);
  main.variable(observer()).define(["showTable","telcoData2"], _31);
  main.variable(observer()).define(["md"], _32);
  main.variable(observer("telcoData3")).define("telcoData3", ["telcoData2"], _telcoData3);
  main.variable(observer()).define(["showTable","telcoData3"], _34);
  main.variable(observer()).define(["md"], _35);
  main.variable(observer("telcoData4")).define("telcoData4", ["telcoData3"], _telcoData4);
  main.variable(observer("paymentMethods")).define("paymentMethods", ["telcoData4"], _paymentMethods);
  main.variable(observer()).define(["md"], _38);
  main.variable(observer()).define(["Plot","telcoData4"], _39);
  main.variable(observer()).define(["md"], _40);
  main.variable(observer()).define(["md"], _41);
  main.variable(observer()).define(["Plot","telcoData4"], _42);
  main.variable(observer()).define(["Plot","telcoData4"], _43);
  main.variable(observer()).define(["Plot","telcoData4"], _44);
  main.variable(observer("dependentsChart")).define("dependentsChart", ["Plot","telcoData4"], _dependentsChart);
  main.variable(observer()).define(["md"], _46);
  main.variable(observer()).define(["md"], _47);
  main.variable(observer("contractsChart")).define("contractsChart", ["Plot","telcoData4"], _contractsChart);
  main.variable(observer("paperlessBillingChart")).define("paperlessBillingChart", ["Plot","telcoData4"], _paperlessBillingChart);
  main.variable(observer("paymentMethodChart")).define("paymentMethodChart", ["Plot","telcoData4"], _paymentMethodChart);
  main.variable(observer()).define(["md"], _51);
  main.variable(observer()).define(["md"], _52);
  main.variable(observer("tenureChart")).define("tenureChart", ["Plot","telcoData4"], _tenureChart);
  main.variable(observer("monthlyChargesChart")).define("monthlyChargesChart", ["Plot","telcoData4"], _monthlyChargesChart);
  main.variable(observer("totalChargesChart")).define("totalChargesChart", ["Plot","telcoData4"], _totalChargesChart);
  main.variable(observer()).define(["md"], _56);
  main.variable(observer()).define(["md"], _57);
  main.variable(observer("phoneServiceChart")).define("phoneServiceChart", ["Plot","telcoData4"], _phoneServiceChart);
  main.variable(observer("multipleLinesChart")).define("multipleLinesChart", ["Plot","telcoData4"], _multipleLinesChart);
  main.variable(observer("internetServiceChart")).define("internetServiceChart", ["Plot","telcoData4"], _internetServiceChart);
  main.variable(observer("onlineSecurityChart")).define("onlineSecurityChart", ["Plot","telcoData4"], _onlineSecurityChart);
  main.variable(observer("onlineBackupChart")).define("onlineBackupChart", ["Plot","telcoData4"], _onlineBackupChart);
  main.variable(observer("deviceProtectionChart")).define("deviceProtectionChart", ["Plot","telcoData4"], _deviceProtectionChart);
  main.variable(observer("techSupportChart")).define("techSupportChart", ["Plot","telcoData4"], _techSupportChart);
  main.variable(observer("streamingTVChart")).define("streamingTVChart", ["Plot","telcoData4"], _streamingTVChart);
  main.variable(observer("streamingMoviesChart")).define("streamingMoviesChart", ["Plot","telcoData4"], _streamingMoviesChart);
  main.variable(observer()).define(["md"], _67);
  main.variable(observer()).define(["md"], _68);
  main.variable(observer()).define(["md"], _69);
  main.variable(observer()).define(["md"], _70);
  main.variable(observer()).define(["showTable","telcoData4"], _71);
  main.variable(observer("telcoLabelEncod")).define("telcoLabelEncod", ["telcoData4"], _telcoLabelEncod);
  main.variable(observer()).define(["showTable","telcoLabelEncod"], _73);
  main.variable(observer()).define(["md"], _74);
  main.variable(observer("telcoDf")).define("telcoDf", ["dfd","telcoLabelEncod"], _telcoDf);
  main.variable(observer("telcoOneHotEncode")).define("telcoOneHotEncode", ["dfd","telcoDf","SimpleData"], _telcoOneHotEncode);
  main.variable(observer()).define(["showTable","telcoOneHotEncode"], _77);
  main.variable(observer()).define(["md"], _78);
  main.variable(observer("normalization")).define("normalization", ["dfd","telcoOneHotEncode","SimpleData"], _normalization);
  main.variable(observer()).define(["showTable","normalization"], _80);
  main.variable(observer()).define(["md"], _81);
  main.variable(observer("data")).define("data", ["dfd","normalization"], _data);
  main.variable(observer()).define(["md"], _83);
  main.variable(observer()).define(["md"], _84);
  main.variable(observer("SimpleData")).define("SimpleData", _SimpleData);
  main.variable(observer()).define(["md"], _86);
  main.variable(observer("dfd")).define("dfd", ["require","print"], _dfd);
  main.variable(observer()).define(["md"], _88);
  main.variable(observer("showChart")).define("showChart", ["html"], _showChart);
  main.variable(observer("showTable")).define("showTable", ["Inputs"], _showTable);
  return main;
}
