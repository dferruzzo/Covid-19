# Script para ler os dados das três cidades
# Autor : Diego Ferruzzo
# Data: 05/08/2020
# ------------------------------------------------------------------------------
# Clear environment
rm(list = ls()) 
#
require("pacman")
p_load(pacman, rio) 

library(lubridate)
library(stringr)

# muda a pasta de trabalho para a pasta onde está o arquivo .R
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
#

# carregando a data do Covid diretamente do site
covid_csv <- read.csv(url("https://raw.githubusercontent.com/seade-R/dados-covid-sp/master/data/dados_covid_sp.csv"),header=T,sep=';',encoding="UTF-8")
head(covid_csv)

# Município de São Paulo
SaoPaulo <- covid_csv[covid_csv$nome_munic == "São Paulo", c(5,6,10,19)]
head(SaoPaulo)
#pop_saopaulo <- mean(covid_csv[covid_csv$nome_munic == "São Paulo",c(19)])
# Município de Santos
Santos <- covid_csv[covid_csv$nome_munic == "Santos", c(5,6,10,19)]
head(Santos)
#pop_santos <- mean(covid_csv[covid_csv$nome_munic == "Santos",c(19)])
#Municipio de Campinas
Campinas <- covid_csv[covid_csv$nome_munic == "Campinas", c(5,6,10,19)]
head(Campinas)
#pop_campinas <- mean(covid_csv[covid_csv$nome_munic == "Campinas",c(19)])
# 
# salvando os dados
write.csv(SaoPaulo,'data/SaoPaulo_dados_covid.csv')
write.csv(Santos,'data/Santos_dados_covid.csv')
write.csv(Campinas,'data/Campinas_dados_covid.csv')

# Carregando a data do indice de isolamento
# Esses dados precisam ser baixados manualmente no site
# https://www.saopaulo.sp.gov.br/coronavirus/isolamento/
isolamento <- import("data/Dados_Full_Data_data.csv",
                     head=T, encoding="UTF-8")
#isolamento <- import("./indice_isol.csv", head=T, encoding="UTF-8")
head(isolamento)

isolamento$Data <- as.Date(str_sub(isolamento$Data,start =-5),"%d/%m")
head(isolamento)
isolamento$`Índice De Isolamento` <- as.numeric(sub("%","",
                              isolamento$`Índice De Isolamento`))/100
head(isolamento)
#
isol_saopaulo <- isolamento[isolamento$Município == "SÃO PAULO",c(3,4)]
head(isol_saopaulo)
# sorting by date
isol_saopaulo <- isol_saopaulo[order(as.Date(isol_saopaulo$Data, format="%Y-%m-%d")),]
head(isol_saopaulo)
summary(isol_saopaulo)
#
isol_santos <- isolamento[isolamento$Município == "SANTOS",c(3,4)]
head(isol_santos)
# sorting by date
isol_santos <- isol_santos[order(as.Date(isol_santos$Data, format="%Y-%m-%d")),]
head(isol_santos)
summary(isol_santos)
#
isol_campinas <- isolamento[isolamento$Município == "CAMPINAS",c(3,4)]
head(isol_campinas)
# sorting by date
isol_campinas <- isol_campinas[order(as.Date(isol_campinas$Data, format="%Y-%m-%d")),]
head(isol_campinas)
summary(isol_campinas)
# -----------------
# salvando os dados
#write.csv(isol_saopaulo,'SaoPaulo_isolamento.csv')
#write.csv(isol_santos,'Santos_isolamento.csv')
#write.csv(isol_campinas,'Campinas_isolamento.csv')
