---
doc_id: ecb_hfcs_wealth_surveys_at_the_ecb
title: Wealth surveys at the ECB
source: ecb_hfcs
source_type: macro_data
url: https://www.ecb.europa.eu/home/pdf/research/hfcn/WealthSurveys.pdf
language: en
geography: EU
topic: wealth
columns:
- Age
- Balance
date: null
retrieved_at: '2026-11-19'
license: 'ECB publications: reuse allowed with source acknowledgement'
format: pdf
---

# Wealth surveys at the ECB

Re-weighting to reduce unit non-response bias in household
wealth surveys: a cross-country comparative perspective
illustrated by a case study
Sébastien Pérez-Duarte, Carlos Sánchez-Muñoz, Veli-Matti Törmälehto1
European Central Bank, Statistics Development and Co-ordination Division
Kaiserstrasse 29
60311 Frankfurt am Main, Germany
E-mail: veli-matti.toermaelehto@ecb.int

Keywords: survey, unit non-response, re-weighting, calibration, wealth, euro area HFCS
Abstract
We discuss the problem of unit non-response in household wealth surveys, and review
possible strategies to reduce non-response bias in the Eurosystem Household Finance and
Consumption Survey (HFCS). The HFCS is a Eurosystem survey coordinated by the European
Central Bank, to gather a set of ex-ante harmonised household wealth surveys from all euro area
countries.
Information on idiosyncratic characteristics of non-respondents may not only be used to
improve the data collection process (e.g. sample design, targeted contact strategies) in future waves,
but may also crucially contribute to the ex-post adjustment of design weights for non-response. In
addition, calibration of sampling weights on external auxiliary information may further help reduce
unit non-response bias. For the specific case of the euro area wealth survey, we discuss how these
two steps (namely non-response adjustment and calibration to auxiliary information) may be used
to reduce non-response bias.
As a case study, we examine unit non-response in the Finnish Wealth Survey 2004 with
information linked from registers to the non-responding part of the sample. We use the Finnish data
to study how properties of selected wealth and debt indicators change with the amount of auxiliary
information used in the calibration adjustments, i.e. from just using widely available variables (such
as age and gender) to using a much larger set of variables (possibly existing only in a few
countries). With these results we intend to extract lessons as to whether different country re-
weighting approaches may have substantial effects on country comparability for the case of a cross-
national survey such as the HFCS.
1. Introduction
This paper discusses how re-weighting may help reduce unit non-response bias in household
wealth surveys by using information on idiosyncratic characteristics of non-respondents as well as
through calibration of sampling weights on external auxiliary information.
While unit non-response may potentially affect any kind of survey, household surveys on
wealth typically register lower response rates largely because of the sensitivity of the survey topic.

1 We would like to thank Mr. Markku Säylä (Statistics Finland) whose help was key in the realisation of this paper as well as
Claudia Biancotti, Jiri Slacalek and Caroline Willeke (ECB) for providing very useful comments.

The views expressed in this paper (as well as any possible mistakes) are exclusively the authors’ and do not necessarily represent
the views of either the European Central Bank nor of the Eurosystem Household Finance and Consumption Network.

- 2 -
In addition, such surveys may severely suffer from the effects of non-random selectivity caused by
the difficulty to contact and convince particular groups of households (e.g. the wealthy) to
participate. First of all, it is well-known that the distribution of wealth across households is highly
skewed, i.e. a relatively small number of wealthy households own an important share of total
household wealth. In addition, typically only a reduced share of households hold certain financial
products, particularly the most sophisticated ones. Consequently, if such households are not well
represented in the final sample, the results may not provide a reliable picture of the distribution and
composition of household portfolios.
Achieving a representative portion of wealthy respondents through standard sampling-design
and contact-strategy procedures may be challenging for several reasons. Firstly, establishing contact
with wealthy respondents may be more difficult as they are likely to possess more than one
residence, to be surrounded by additional security measures and also to be absent during prolonged
periods of time. Secondly, both available time as well as self-perceived value/time ratio usually pre-
dispose wealthy households to refuse participating in surveys; in practice, a differentiated contact
and interviewing strategy possibly implying additional persuasion effort is required. In many of
these cases, best-practice measures may help mitigate the effects on final non-response, i.e. more
flexible contact strategies and adaptation to household schedules, additional contact efforts,
collection of information from non-household members, etc. 2
Mitigating strategies notwithstanding, unit non-response is unlikely to be a random
phenomenon and is likely to more severely affect wealthy households, which also happen to be the
ones potentially providing the most interesting information in the specific case of household wealth
surveys.
This is an important challenge for the Eurosystem Household Finance and Consumption
Survey (HFCS). The HFCS is an initiative of the euro area national central banks (NCBs)
coordinated by the European Central Bank, which consists of a set of ex-ante harmonised wealth
surveys conducted across all sixteen euro area countries. The HFCS will collect microdata on
household real and financial assets, debts, consumption and saving, income and employment,
demographics, future pension entitlements, intergenerational transfers and gifts and risk attitudes.
The resulting micro data will be made available to the research community in the future.3
The paper is split into five sections. Further to this introduction, the second section analyses
differences in the response rates of wealth and other household surveys across countries. The third
section presents, from the perspective of the HFCS as a cross-national survey, ways to maximise
participation as well as how weighting adjustments can help reduce non-response bias,
distinguishing between non-response and calibration adjustments. The fourth section illustrates
possible strategies to reduce non-response bias with a case study based on the whole eligible sample
of the Finnish Wealth Survey 2004, for which substantial auxiliary information from the sampling
frame (i.e. also including non-respondents) and from record linkage to administrative and statistical
registers is available. Using external information extracted from registers, the case study tries to
show how different strategies to calibrate survey results with the use of a more or less exhaustive
set of external auxiliary variables may affect the results. The fifth section concludes.

2 Household Finance and Consumption Network (2008.a).
3 More information on the HFCS is available on http://www.ecb.europa.eu/home/html/researcher_hfcn.en.html

- 3 -
2. Unit non-response in wealth surveys
Substantial differences can be observed in the patterns of unit non-response across different
sub-population groups, but also across different countries. Dissimilar societies, customs and cultural
setups may result in substantially varying attitudes towards survey participation.
To illustrate this point with actual data, Table 1 below compares the response rates registered
in different countries conducting similar wealth surveys as well as other European surveys.
Table 1. Response rates in wealth and other household surveys 4 5

Wealth surveys
EU-SILC 2006
HBS 2005
ESS 2006
Germany
40%
68%
Quota sample
55%
Greece
52%
79%
60%
-
Spain
41%
58%
65%
66%
France
74%
71%
57%
46%
Italy
42%
82%
75%
-
Austria
65%
59%
42%
64%
Portugal
57%
82%
62%
73%
Finland
66%
75%
52%
64%
United Kingdom
55%
77%
57%
54%
United States
44%

Column 1 shows quite some degree of variation across countries concerning response rates
for wealth surveys, ranging from around 40% for the surveys conducted in Germany, Spain, Italy
and the US to 74% for the French survey. In addition to the above-mentioned cultural and social
differences, this degree of variation can also be explained by a number of factors such as (i)
whether each specific survey over-samples wealthy households (substantially increasing sampling
efficiency at the expense of increasing the overall level of (unweighted) non-response, as is the case
of the Spanish and US surveys); (ii) survey exhaustiveness (for instance the Austrian survey only
covers housing – but not financial – wealth, thus decreasing the number of sensitive topics in the
interview, with a positive effect in response rates); (iii) whether survey participation is legally
binding to respondents (as may be the case of surveys run by National Statistical Institutes); (iv)
sample design and contact strategies (whether the sample design is based on registers of
households/persons or of dwellings, with the former approach permitting a more personalised first-
contact strategy positively impacting on response rates); (vi) allowing proxy respondents; (v)
making use of non-respondent substitutes/replacements, etc.
It is also possible to horizontally compare response rates within countries between wealth and
other kinds of surveys, namely the Survey on Income and Living Conditions (SILC), the Household
Budget Survey (HBS) and the European Social Survey (ESS). As can be seen, there is a tendency in
each country toward lower response rates in wealth surveys compared with those in other surveys
(with the exception of France and Austria). This is a logical consequence of the sensitiveness of the
survey topic, the length of the interviews (highly dependent on the complexity of the household’s

4 In making comparisons across countries, it is very important to dispose of comparable figures, namely to apply a single
methodology for the calculation of response rates. In particular, Table 1 was elaborated by computing response rates as the ratio of
final interviews to total eligible households in the sample.
5 Wealth surveys: DE: Socio-Economic Panel (SOEP) 2006; ES: Encuesta Financiera de las Familias 2005; FI: Wealth and
Housing Survey 2004; FR: Enquête Patrimoine 2004; GR: Survey of Indebtedness of Greek Households 2007; IT: Survey of
Household Income and Wealth 2006; PT: Inquérito ao Património e Endividamento das Famílias 2006; US: Survey on Consumer
Finances 2007; UK: Wealth and Assets Survey 2006/08.

Other household surveys: EU-SILC: Survey on Income and Living Conditions; HBS: Household Budget Survey; ESS: European
Social Survey.

- 4 -
portfolio) as well as the level of effort required from them (mental calculations, preparation and use
of supporting documentation, etc.).
To confirm the validity of these calculations, in particular with a view to double-checking the
comparability of the country figures, it may be useful to dig more into the specifics of how response
rates are calculated for each survey. Table 2 below shows the gross sample size for each survey, the
number of units regarded as ineligible, those for which it was not possible to establish contact, and
finally the split between refusals and other reasons for non-response (language barriers, discarded
after supervision, discarded due to high item non-response, etc.).
Table 2. Response rate components in wealth surveys

Sample
Out of
scope/over-
coverage
Eligible cases
(1) Non-
contact
(2) Total
Refusal
(3) Other
reasons for
non-response
Total
Respondents
3931
181
3750
485
1659
100
1506
Germany (2006)
Socio-Economic Panel (SOEP)
105%
5%
100%
13%
44%
3%
40%
15662
1275
14387
1602
6823
189
5962
Spain (2005)
EFF
109%
9%
100%
11%
47%
1%
41%
15093
1939
13154

9692
France (2004)
Enquête Patrimoine
115%
15%
100%

74%
18814
304
18510
4139
6603

7768
Italy (2006)
SHIW
102%
2%
100%
22%
36%

42%
3338
141
3197
433
636
47
2081
Austria (2008)
OeNB Survey on Wealth
104%
4%
100%
14%
20%
1%
65%
5300
74
5226
338
1400
33
3455
Finland (2004)
Household Wealth Survey
101%
1%
100%
6%
27%
1%
66%
62800
6971
55829
4135
19210
1889
30595
UK (2006/08)
Wealth and assets survey
112%
12%
100%
7%
34%
3%
55%
9980

9980
288
5114
156
4422
US (2007)
SCF
100%

100%
3%
51%
2%
44%
Note: Other reasons for nonresponse includes language barriers, high item non-response, items discarded after
supervision, etc. Blank cells correspond to information not available to the authors.
Sources: following publications plus authors’ own calculations: Finland: Niemeläinen, S., T. Sandström, M. Säylä, V.-
M. Törmälehto (2006) / Spain: Bover, O. (2008) / Italy: Banca d’Italia (2008) / Austria: Fessler, P., P. Mooslechner, M.
Schürz and K. Wagner (2009) / Germany: Von Rosenbladt, B., N. Sigel, S. Stimmel and F. Strutz (2007) / France:
INSEE (2007) / US: Kennickell, A. B. (2009) / UK: Office for National Statistics (2009)
Mirroring the substantial degree of variation across countries observed in Table 1, a sizeable
divergence also exists in the distribution of eligible cases between non-contacts, refusals, other non-
response and final respondents. For instance, the non-contact rate is particularly high for the Italian
SHIW, while it is fairly low in the case of the Finnish, the US and the UK surveys. This may have
to do with the sampling approach as well as with the level of effort employed in each individual
case, which reduces the effects on selectivity bias at the expense of increasing total survey costs per
individual respondent.
With regard to refusals, the highest rates are registered in the case of the wealth surveys
corresponding to US, Germany and Spain, while the lowest rates correspond to the surveys
conducted in Austria (possibly related to the survey focus limited to housing wealth) and Finland

- 5 -
(where the survey is conducted by the National Statistical Institute6). The split between direct
refusals and other non-response (due to language barriers, interviews that had to be discarded due
to high item non-response, etc.) is quite homogenous, ranging in between 1% and 3% in all cases.
It would obviously be very interesting to know more about the specific reasons why potential
respondents directly refused to participate in order to consider alternative contact strategies with a
view to future survey waves. Further information is only available for the Finnish survey though7.
3. Maximising participation and reducing non-response bias in cross-national surveys: the
HFCS
3.1 Approaches to reduce non-response for the HFCS
The figures analysed so far suggest that different approaches and different practices
significantly affect response rates across countries and across surveys. Indeed, while some issues
cannot be controlled (e.g. the survey theme and its effects on potential respondents, the need to
cover both urban and rural areas - with response rates significantly lower in the former -, etc.),
applying best practices with regard to some others may make a big difference. During the
preparatory work for the Eurosystem HFCS, different practices were carefully studied and
recommendations to minimise non-response were put forward.8
The following is a non-exhaustive selection of factors that were identified as playing a crucial
role (and on which Member States were advised to put special emphasis): (i) selecting experienced
interviewers and participating actively in their training; (ii) carefully monitoring interviewers’
workload and compensation scheme; (iii) providing concrete instructions as to the contact
schedule;9 (iv) taking due care in providing information materials beforehand (e.g. advance letter
signed by NCB Governor closely before the first interviewer’s visit, supporting materials to be used
during on-site visits covering uses of the data, etc.); (v) selecting incentives that respondents may
particularly appreciate;10 and (vi) providing feedback to respondents ex-post about uses of the data
(particularly helpful in the case of panels).
As a last resort, refusal conversion (i.e. applying additional effort to convince initially
reluctant respondents to participate, possibly via a refusal-conversion specialist) may also have non-
negligible effects on response rates.
3.2 Re-weighting as a way to reduce unit non-response bias
Regardless of the preventive measures taken, response rates in the euro area wealth surveys
may be lower than in other household surveys such as those included in Table 1 most likely due to
the sensitivity of the survey topic. Lower response rates imply lower efficiency of the estimates
because of smaller effective sample sizes. Whether the estimates are more biased depends on the

6 NSIs’ reputation may make them more trustworthy vis-à-vis respondents than it may be the case of private survey companies.
7 Out of 1,400 refusals in the Finnish survey, 160 were due to the survey topic, 276 because of alleged lack of time, 274 were
hard-core non-respondents (units which as a matter of principle do not participate in any kind of surveys), and 403 did not give any
specific reason.
8 Household Finance and Consumption Network (2008.b).
9 It was specifically recommended to make a minimum of four personal visits to each sampling unit (including at least one visit in
the evening and one over the weekend, spread over at least two different weeks) as well as to systematically record contact attempts
and check them regularly.
10 The best incentives very much vary from country to country: symbolic tokens (for instance symbolic goods only produced by
central banks), lottery tickets, charity donations, occasionally some money, etc.

- 6 -
(lack of) of non-response randomness and on the extent that re-weighting of design weights may
help reduce the possible bias.
The weights should carry information on both the sample selection process (design weights)
and non-response process (labelled final estimation weights in the HFCS). The cross-sectional
weighting in the euro area HFCS is expected to follow the standard two-step path, i.e. (1)
adjustment of the design weights for non-response (sample-based non-response weighting) and (2)
adjusting the weights such that particular estimates will match external information (population-
based adjustment of weights).
3.2.1 Re-weighting to adjust for non-response in the HFCS
Adjustment of the design weights for non-response may be conducted by using response
propensities, for instance within homogenous response groups. This adjustment requires that
information on the characteristics of non-respondents are known.
Generally, in the 16 euro area countries the following data may be available, to a varying
extent, to assess the characteristics of the non-respondents (largely only available at country level):
1) Contact attempt information
2) Interviewer observations on non-respondents (paradata information)
3) Panel attrition in subsequent waves (for those surveys with a longitudinal component)
4) Microgeographic information
5) Information derived from the sampling frame
6) Record-linked data from administrative and statistical registers (only for a few surveys)
In the euro area survey, (1) contact attempt information and (2) interviewer observations
should be centrally available as countries will in principle report them to the ECB. Survey process
data such as contact attempts may be used in the non-response adjustment by assuming that non-
respondents have similar characteristics as those households who have responded only after several
contact attempts.11
Interviewer observations on non-respondents in the euro area HFCS comprise descriptive
information on geographical distribution, dwelling rating, appearance, location, surrounding
neighbourhood, and security measures which are part of the paradata information collected in the
survey. Such variables may be very useful to detect specific patterns of non-response. In addition,
those non-respondents who refuse to participate could ideally also be convinced to provide some
limited information on personal reasons to refuse participating.
Some surveys (e.g. those in Italy, Spain, Germany, the Netherlands and Belgium) include a
longitudinal component and thus in the coming years characteristics from previous waves could be
used to study panel attrition and to estimate response propensities, if it is assumed that panel
attrition can be used to make inference on cross-sectional response propensities.
Microgeographic information (available at country level but not centrally collected by the
ECB for confidentiality reasons) may help if for instance the income level of the local
neighbourhood (e.g. at zip-code level) where the non-responding household lives is known although
the income level of the non-responding household itself is not available.
Exact matching on case level is generally not feasible in the euro area countries, but some
existing surveys have been able to use register-based auxiliary information in sampling and non-
response analysis. The Spanish Wealth Survey (EFF) has used wealth tax data, both to stratify the
sample to over-sample wealthy households and to describe unit non-response. The Finnish

11 For example, D’Alessio and Faiella (2002) examine response propensities in the Italian wealth survey SHIW by assuming that
households requiring two or more attempts are representative of non-responding units as a whole.

- 7 -
Household Wealth Survey extensively uses administrative data in stratification, estimation and also
as survey outcome variables (see case study in the third section of this paper).
3.2.2 Calibration adjustments in the HFCS
Adjustments to external sources require that information for each specific respondent is
available in an external source at population level. The adjustments should build on non-response
analysis but it is likely that in practise they highly depend on the information available to each
country. Typically, in the existing euro area wealth surveys the weights have been adjusted to
demographic and geographical distributions such as age, gender, household size, geographical area,
and size of the geographical area (Table 3). Such information is usually available from the census
or other major household surveys and the necessary condition of strict correspondence of variable
definition in the sample and the external source is generally satisfied. The HFCS definitions are
aimed to be aligned with, for example, the core social variables defined by Eurostat. This should
improve also possibilities for using information derived from other surveys in the weight
adjustments.
Table 3. Overview of external information used for weight adjustments in some existing
household wealth surveys.12
Country and survey
External information used in re-weighting
Germany 2006
Socio-Economic Panel (SOEP)
Age, gender, non-German nationality, region, community
size, household size, home ownership status (Kroh, 2009)
Spain 2005(*)
Encuesta Financiera de las Familias
Gender, age by municipality size, and household size by
municipality size (Bover, 2008)
France 2004(*)
Enquête Patrimoine
Age, socioeconomic status and education of the reference
person, gender, household type and region (INSEE 2007)
Italy 2006(*)
Survey of Household Income and Wealth
Gender, age group, geographical area and size of the
municipality of residence (Banca d’Italia 2008)
Austria 2008(*)
Household Survey on Housing Wealth
Province of residence, size of municipality, size of household
(Fessler et. al. 2009)
Finland 2004(*)
Household Wealth Survey
Household size, age and gender, region, degree of
urbanisation, income totals, number of income recipients,
total amount of taxable wealth (Statistics Finland 2007)
United Kingdom 2006/08
Wealth and assets survey
Age and sex, region (ONS, 2009)
4. A case study with Finnish data
In this section, we study non-response bias and alternative re-weighting scenarios using the
Finnish Household Wealth Survey 2004 which had an overall unweighted non-response rate of 34
percent. This wealth survey data set is particularly well suited for bias analysis because variables
from administrative and statistical registers can be record-linked to the sample, including variables
on taxable wealth, outstanding debt, and income.13 Therefore, for all members of the sample we
have identical measurements which are good proxies (but still only proxies) of the eventual survey
variables. Finally, the Finnish Wealth Survey 2004 by and large complies with the requirements set

12 Surveys marked with an asterisk are part of the HFCS.
13 The wealth tax was abolished in Finland in 2006, so this is the last wealth survey data with the variable on taxable wealth
available. The wealth tax has also been abolished in Austria, Germany, Luxembourg, the Netherlands, and Spain in recent years,
leaving France and Greece as the only two euro area countries with a tax on net worth.

- 8 -
for the euro area HFCS14, such that the results may therefore complement the existing knowledge
about non-response in wealth surveys.
The Wealth Survey 2004 sample is a stratified one-stage probability proportional-to-size
sample, drawn in two phases. In the first phase, persons are drawn from the population register to
create a fairly large master sample. Because the frame is sorted by domicile code (address),
sampling is proportional to household size and there is implicit stratification by region. In the
second phase, the final sample is drawn from the master sample, stratifying explicitly by type and
level of income. The explicit stratification over-samples high-income households but in terms of
wealth, the over-sampling is somewhat limited. Table 4 compares the over-sampling rate15 in the
Finnish survey with the Spanish Wealth Survey (EFF 2005) and the Italian Wealth Survey (SHIW
2006)16.
Table 4. The degree of oversampling in the final sample: Spain, Italy and Finland

Net worth quantiles

Bottom 50 % 50 % to 90 %
90 % to 95 %
95 % to 99 %
Top 1 %
Sample
size, net
Spain
0.75
0.85
1.61
2.83
8.99
5,962
Italy
0.94
1.06
1.08
1.07
1.16
7,768
Finland
0.83
1.07
1.49
1.67
1.59
3,455
Source: Bover (2008), Statistics Finland (2007), SHIW data 2006; authors’ own calculations.
The permanent field interviewers of Statistics Finland attempted to contact the person
selected from the population register. In this paper, the household characteristics from the registers
are characteristics of this selected person. In responding households, the survey responses were
sought from the person best aware of the financial situation of the household.
4.1 Non-respondents in the Finnish Household Wealth Survey 2004
The basic descriptive breakdowns of non-respondents are included in Statistics Finland
(2007), showing that non-response is higher in urban areas, in Southern Finland, among less
educated households, and among middle income households. Remarkably, and contrary to what one
might expect, non-response rates did not increase along the level of taxable wealth. This may be a
survey-specific feature as response rates in the top decile of taxable wealth increased substantially
compared with the previous 1998 survey.
In Table 5 we provide results of a model-based analysis of the determinants of non-response.
One motivation is finding the variables most correlated with response propensities so that the re-
weighting scheme could be designed to reduce the ensuing no-response bias.
Overall the (weighted) non-response rate is 33%, but there is significant variation across
strata, age groups, regions, and education levels; women are also slightly more likely to participate.
In particular, “farmers” (a category determined from the tax register) have a much lower than
expected probability of not responding, while city dwellers, as is common in other countries, have a
higher than usual probability of not responding. Persons whose main language is neither Finnish
nor Swedish (the two official languages in Finland) participate in the survey less often. Although

14 The mode of collection is CAPI, survey has a representative probability sample stratified to over-sample high income
households, and covers assets, liabilities, income, demographics, and employment.
15 The oversampling rate is the number of responding households in the wealth range divided by the number of households one
would expect if the sample was randomly drawn from the population (Bover, 2008).
16 For the SHIW, wealthy households are not over-sampled.

- 9 -
not properly significant, other things being equal the propensity to respond increases with the
education level.
Table 5. Logit regression of non-response on personal characteristics
Variable17
Odds ratio
t-stat

Variable
Odds ratio
t-stat
Stratum 5 (self-employed 1)
0.717*
-1.71

Language:
Swedish
1.080
0.47
Stratum 6 (self-employed 2)
0.615*
-1.92

Language: Other
1.640**
2.31
Stratum 7 (farmers 1)
0.377***
-3.49

ISCED 1
1.269**
2.51
Stratum 11 (other income 1)
0.773**
-2.19

ISCED 5
0.930
-0.57
Helsinki
1.368***
2.96

ISCED 6
0.797
-1.38
Helsinki region
1.528***
3.51

ISCED 7
0.794
-1.44
Other main cities
1.276**
1.96

ISCED 8
0.568
-1.28
Male
1.129
1.49

Debt decile 5
0.914
-0.69
Age: 16-24
0.438***
-5.06

Debt d6
0.672***
-2.92
25-34
0.750**
-2.07

Debt d7
0.838
-1.23
35-44
1.108
0.82

Debt d8
0.714**
-2.23
55-64
0.858
-1.15

Debt d9
0.849
-1.05
65-74
0.712
-1.54

Dividend decile 7
0.813
-1.16
75+
0.578***
-2.29

Dividend d8
0.672**
-2.20

Dividend d9
0.670**
-2.20
Source: Statistics Finland, Wealth Survey 2004. Authors’ calculations. Logit regression of non response on other
variables; out of scope persons were excluded; design weights were used. An odds-ratio above one indicates that
persons with this characteristic have a higher probability of not responding, other things being equal; since all variables
in this table are indicator variables, marginal effects can be computed by taking the natural logarithm of the odds ratio.
Number of obs.: 5,155. The sign * (respectively **, ***) indicates that the coefficient is significant at the 10% level
(respectively, 5%, 1%). Omitted category: Households in stratum 1 (households with wages and salaries), in rural areas,
whose selected person is female, 45 to 54 years old, speaks Finnish, has education level ISCED 3, no debt and receives
no dividends. Additional variables not presented in this table and not significant: other strata, household size, and
deciles for total income, wealth, capital income and other income.

The only wealth-related variables to be significant in the logit regression are indebtedness and
the existence of dividend income; ceteris paribus, indebted households and those receiving some
kind of dividend income tend to participate more in the survey. However, the logit regression is
only marginally successful in explaining non-response, as the percentage of variance explained is
low.
The propensity score, calculated as the predicted probability of participating in the survey, is
shown in Figure 1 for the respondents and the non-respondents. The support of this distribution is
[0.1;0.8], and the propensity score of the respondents strictly dominates the score of non
respondents at all levels. Although this propensity score could be used to adjust the design weights
to compensate for non-response, in practice this approach may not be that common in the HFCS
surveys. Therefore, in this paper the weights are only adjusted on the basis of homogenous response
groups.

17 Stratification: 1-4 = wages and salaries (four income categories); 5-6 = self-employed; 7-8 = farmers; 9-10 = pensioners; 11-12 =
other income sources

- 10 -
0
1
2
3
4
Density
0
.2
.4
.6
.8
Predicted probability of non-response
Respondents
Non-respondents
Figure 1: Kernel density estimate of the predicted probability of
not responding

4.2 Non-response bias in sample means of income, debt and wealth
Within a survey, different estimates may have variation in their unit non-response bias
depending on the correlation between variables and response probabilities. Some of the important
outputs of the euro area survey are expected to be debt, wealth and income levels as well as their
ratios (leverage ratios) by population subgroups. Table 6 shows the means of register-based
disposable income, debt, and taxable wealth18 for respondents, the difference between respondents
and non-respondents, and the relative difference with respect to design-weighted estimates of the
whole eligible sample, which is labelled here as “relative bias”.
Table 6. Disposable income, debt and taxable wealth: weighted means (euro / household) for
respondents and non-respondents; relative bias due to unit non-response
Age
Non-
response
Disposable income
Debt
Wealth

Weighted
share
R
R-NR
Relative
bias
R
R-NR
Relative
bias
R
R-NR
Relative
bias
All
34%
28,913
-680
-1% 14,119
477
1% 26,784
2388
3%
16-24
30%
25,096 -5,265**
-6% 4,949 2,391**
15% 2,510
-108
-1%
25-34
32%
30,122
2,796**
3% 27,824 5,358**
6% 14,346 3,529**
8%
35-49
38%
35,847
93
0% 25,632
1615
2% 27,166
2555
4%
50-64
34%
32,169
1099
1% 9,931
947
3% 39,400
5341
5%
65-
31%
19,563
-114
0% 1,419
-499
-11% 35,178
2813
2%
Source: Statistics Finland Wealth Survey 2004, authors’ calculations.
Note: R = respondents, NR = non-respondents. ** = difference statistically significant at 5 % level. Weights: Design
weights.

18 In terms of levels, taxable wealth is not a good proxy of household wealth because of undervaluation of assets in taxation and
imperfect coverage of certain asset types (e.g. savings in interest-bearing accounts). The correlation with the survey variables is
strongly positive, however. Register-based debts seem to underestimate somewhat the debts recorded in the survey while disposable
income is a near perfect proxy of household disposable cash income.

- 11 -
Overall, there appears to be a small downward bias in disposable income and upward bias in
debt and wealth levels due to unit non-response, but the differences between respondents and non-
respondents are not statistically significant. In the age group 25-34, the differences are statistically
significant and there is an upward bias in means of all three variables. In the age group 16-24, we
see a downward bias in income and upward bias in liabilities. Note that these age groups have in
fact higher response rates than other groups so the bias derives from differences between those who
respond and those who do not, i.e. not from non-response levels but rather from non-response
selectivity. For the other age groups, the hypothesis of equal means of responding and non-
responding households cannot be rejected.
4.3 Alternative re-weighting scenarios and non-response bias in the Finnish survey
The table above used design weights for the whole eligible selected sample. We now limit
ourselves to register-based debt and compute estimates with alternative weights for the respondents.
The reduction in non-response bias due to re-weighting is evaluated as the difference between
estimates obtained for the responding sample from those obtained with design weights for the
whole eligible sample.
Specifically, we compare the indicators across the different weighting scenarios shown in
Table 7.19
The first calibration uses basic population level variables that seem to be available in many
countries: age and gender, region and household size. This model is here taken to represent the
likely minimum set of common variables which potentially could be used in all euro area countries.
The second calibration model extends the information set by bringing in education (ISCED of
selected respondent), language (Finnish/Swedish/Other), share of indebted households, and total
amounts of income, debt, and taxable wealth.
Table 7. Re-weighting scenarios used in this study
Weight
Construction
Weights adjusted for non-
response at sample level
Non-response correction with within strata response rates (Homogeneous
response groups)
Calibrated weights I,
“euro area” variables
Age (5-year bands) and gender, region/density (4 categories), household size
(6 categories)
Calibrated weights II,
extended calibration
Age (5-year bands) and gender, region/density (4), household size (6),
education (6), language (3), indebtedness (2), total amount of taxable wealth,
total amount of debt, total amount of income

The key question is whether there are substantial differences between the basic “euro area”-
model (age and gender, household size, region) and the more extended model – that is, whether the
amount of auxiliary information could have some bearing for cross-country comparability of the
estimates.
In the re-weighting, the weights are calibrated to marginal distributions derived from the
whole eligible sample using the design weights. The aim here is to show whether different
information sets help reduce non-response bias. For this reason and because many variables exist
for non-respondents as well, taking margins from the design-weighted sample seems a more

19 The actual weights used in the HWS 2004 use information from three sources: master sample (household size), another survey
(age of household head and total number of households from EU-SILC), and administrative and statistical registers (total amount of
income and number of recipients for several types of income, total amount of taxable wealth).

- 12 -
transparent choice. In reality, the weights have been calibrated to marginal distributions derived
from external sources.20
Table 8 presents the results, showing relative bias and estimated standard errors21 of means
for the re-weighting schemes.
Table 8: Impact of re-weighting on register-based debt, relative bias and standard errors

Design weights,
unadjusted
HRG adjusted
weights
Calibration: "euro
area" variables
Calibration:
extended model

Mean,
eligible
sample
Relative
bias,
respondents
Relative
bias
Standard
error, %
Relative
bias
Standard
error, %
Relative
bias
Standard
error, %
All
13,835
1.3%
0.6%
3.4%
1.9%
3.2%
0.0%
0.0%
16-24
4,110
13.6%
13.3%
16.7%
14.2%
16.9%
10.3%
16.2%
25-34
26,090
6.2%
5.8%
5.9%
6.4%
5.9%
6.1%
5.0%
35-49
24,899
2.7%
2.0%
5.1%
0.8%
4.9%
-0.3%
3.8%
50-64
9,559
3.4%
3.2%
7.8%
5.4%
8.0%
1.5%
7.7%
65+
1,561
-10.5%
-11.9%
17.3%
-10.9%
17.6%
-19.0%
17.6%
Properties of sample and weights
Sample size
5,226
3,455
3,455
3,455

3,455
Sum of weights 2,387,166
1,592,592
2,455,425
2,387,166
2,387,166
CV of weights
74.1
74.1
75.1
78.0

79.2
The results are not particularly encouraging in terms of removing unit non-response bias
observed in the age groups 16-24 and 24-34 years. Simple adjustment with within-strata response
rates seems to perform even better than calibration to basic key variables; calibration to the “euro
area” variables does not improve the estimates and does not bring any efficiency gains in this
sample. Likewise, the extended model is unable to remove the bias within the age groups although
it does perform somewhat better than the other weighting alternatives.
The extended calibration uses total debt amount and household size distribution in the
calibration. Consequently, by definition, there is neither bias nor variance in the estimate of mean
debt for all households – with this re-weighting scheme the results would always be the same,
irrespective of the sample at hand. Within age groups, the gains in efficiency are considerably
smaller.
5. Conclusions
Given the importance of wealthy households for the analysis of the wealth distribution and
their overall lower tendency to participate in surveys, non-response in wealth surveys is an issue
that needs to be carefully addressed. We have discussed how good interviewing practices can be put
in place during the fieldwork to increase participation. The effectiveness of such practical
recommendations may be limited though in the absence of an external source of micro-level
information on wealth supporting the sample design (at a moment in which wealth taxes are being
abolished in most European countries) that could permit adopting differentiated contact strategies
vis-à-vis different household groups (e.g. the wealthy).
After data have been collected, the impact of non-response can be reduced by re-weighting
the sample in order to correct for the differential response rates across sample frame categories or

20 Re-weighting was done with Calmar 2 SAS-macro using the raking ratio method (M=2).
21 Standard errors are based on 1000 bootstrap re-samples. Weights of every resample were calibrated according to the appropriate
calibration model.

- 13 -
by calibrating on external information. In the context of the Eurosystem Household Finance and
Consumption Survey, this external information is likely to be limited to age, gender, household
size, and some geographical information, and it is unclear whether this limited information will be
sufficient to correct for non-response bias affecting wealth variables.
To provide an insight, we use data from the Finnish Wealth Survey 2004, which contains
register data on income, taxable wealth, debts and other personal data, and analyse non-response
process and different re-weighting strategies. The key messages from this case study can be
summarised as follows. First, in this sample and for these indicators, and in contrast with results
from other wealth surveys, unit non-response did not increase with wealth, after controlling for
demographic characteristics. Second, non-response bias was perhaps smaller than expected; in the
case of people below 35 years, there is a statistically significant non-response bias in income and
debt even though non-response rates are actually lower for this age group. Third, the standard
adjustment procedure, i.e. making first naïve adjustment at sample level and then calibrating to
population margins, did not do much to reduce the non-response bias observed in certain age
groups. Finally, the different calibration approaches shown here using different levels of auxiliary
information did not bring forward sizeable differences. For the specific case of a cross-national
survey such as the HFCS, these results may suggest that different country approaches as to the
number of variables used for calibration may not have substantial effects on cross-country
comparability.
Re-weighting in the case study did not lead to a significant non-response bias reduction,
However, the exercise itself was only possible because register information was available that
enabled measuring non-response bias and analysing possible adjustments. In turn, this paper
illustrates the value of maximising the amount of information available beforehand to design the
sample as well as of disposing of accurate external information to calibrate survey results ex-post.
Such supporting information ultimately become essential for assessing the relevance and the
precision of the collected data, and this may hold especially true in the context of a cross-national
survey such as the HFCS.
REFERENCES
Bover, O. (2008) “The Spanish Survey of Household Finances: Description and methods of the
2005 Wave”, Occasional paper No 803, Banco de España

http://bankofspain.net/informes/be/ocasional/do0803e.pdf
Banca d’Italia (2008) “Household Income and Wealth in 2006”; Supplements to the Statistical
Bulletin, Sample Surveys, Volume XVIII Number 7, 28 January 2008
D’Alessio, G. and I. Faiella, “Non-response behaviour in the Bank of Italy’s and Wealth”, Temi di
discussione, Banca d’Italia n°462, 2002
Fessler, P., P. Mooslechner, M. Schürz and K. Wagner (2009) “Housing Wealth of Austrian
Households”, In: Monetary Policy & the Economy Q2/09. Vienna: OeNB. 104–124.

http://www.oenb.at/en/img/mop_2009_q2_analyses05_tcm16-141274.pdf
Household Finance and Consumption Network (2008.a) “ Oversampling the wealthy: eye for an
eye, euro for a euro”, supplementary document 1.b of the HFCN report to the Governing Council
http://www.ecb.europa.eu/home/pdf/research/hfcn/Oversampling.pdf?f48e61056501e86404f80dcd0
298d020
Household Finance and Consumption Network (2008.b) “Reducing non-response bias”,
supplementary document 1.c of the HFCN report to the Governing Council

http://www.ecb.europa.eu/home/pdf/research/hfcn/Nonresponse.pdf

- 14 -
INSEE (2007), La méthodologie de l'enquête Patrimoine 2004

http://www.insee.fr/fr/methodes/sources/pdf/Patrimoine.pdf
Kennickell, A. B. (2009), “Getting to the Top: Reaching Wealthy Respondents in the SCF”, Paper
prepared for the 2009 Joint Statistical Meetings, Washington, DC

http://www.federalreserve.gov/Pubs/oss/oss2/papers/asa2005.5.pdf
Niemeläinen, S., T. Sandström, M. Säylä, V.-M. Törmälehto (2006) "Housing Wealth in Finland",
29th General Conference of The International Association for Research in Income and Wealth,
Joensuu, Finland, August 20 -26, 2006 http://www.iariw.org/papers/2006/sami_paper.pdf
Office for National Statistics (2009), “Wealth in Great Britain: Main Results from the Wealth and
Assets Survey 2006/08” http://www.statistics.gov.uk/downloads/theme_economy/wealth-assets-
2006-2008/Wealth_in_GB_2006_2008.pdf
Von Rosenbladt, B., N. Sigel, S. Stimmel and F. Strutz (2007) “SOEP 2006 Erstbefragung der
Ergänzungsstichprobe H.”, Methodenbericht TNS Infratest. June.

http://www.diw-berlin.de/documents/dokumentenarchiv/17/83169/meth_2006_h.pdf
Kroh, Martin (2009):“Short-Documentation of the Update of the SOEP-Weights. 1984-2008”.
http://www.diw.de/documents/dokumentenarchiv/17/diw_01.c.342852.de/soep_weights_shortdocu.
pdf
Statistics Finland (2007): “Kotitalouksien varallisuus 1998-2004”. (Household Wealth 1998-2004,
in Finnish with English summary). Official Statistics of Finland, Income and Consumption.
