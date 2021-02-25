# Script para ler os dados das três cidades
# Autor : Diego Ferruzzo
# Data: 05/08/2020
# ------------------------------------------------------------------------------
# Clear environment
rm(list = ls()) 
#
#install.packages("pacman")
require("pacman")
p_load(pacman, rio) 

library(lubridate)
library(stringr)
library(zoo)

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
isolamento <- import("data/Dados_Dados_completos_data.csv",
                     head=T, encoding="UTF-8")
#isolamento <- import("./indice_isol.csv", head=T, encoding="UTF-8")
head(isolamento)

isolamento$Data <- as.Date(str_sub(isolamento$STR_DATA,star=25L,end=-2L),"%d/%m/%Y")
head(isolamento)

isolamento$`?ndice De Isolamento` <- as.numeric(sub("%","",
                              isolamento$`?ndice De Isolamento`))/100
names(isolamento)[names(isolamento) == "?ndice De Isolamento"] <- "Isol"
head(isolamento)
#
isol_saopaulo <- isolamento[isolamento$Munic?pio == "S?O PAULO",c(3,9)]
head(isol_saopaulo)
# sorting by date
isol_saopaulo <- isol_saopaulo[order(as.Date(isol_saopaulo$Data,
                                             format="%Y-%m-%d")),]
isol_saopaulo <- na.omit(isol_saopaulo)
head(isol_saopaulo)
summary(isol_saopaulo)
#
isol_santos <- isolamento[isolamento$Munic?pio == "SANTOS",c(3,9)]
head(isol_santos)
# sorting by date
isol_santos <- isol_santos[order(as.Date(isol_santos$Data, format="%Y-%m-%d")),]
isol_santos <- na.omit(isol_santos)
head(isol_santos)
summary(isol_santos)
#
isol_campinas <- isolamento[isolamento$Munic?pio == "CAMPINAS",c(3,9)]
head(isol_campinas)
# sorting by date
isol_campinas <- isol_campinas[order(as.Date(isol_campinas$Data,
                                             format="%Y-%m-%d")),]
isol_campinas<-na.omit(isol_campinas)
head(isol_campinas)
summary(isol_campinas)
#
# Verificando se há dados faltando
date_range_1 <- seq(min(isol_saopaulo$Data),max(isol_saopaulo$Data), by=1)
missing_dates_1<-date_range_1[!date_range_1 %in% isol_saopaulo$Data]
#
date_range_2 <- seq(min(isol_santos$Data),max(isol_santos$Data), by=1)
missing_dates_2<-date_range_2[!date_range_2 %in% isol_santos$Data]
#
date_range_3 <- seq(min(isol_campinas$Data),max(isol_campinas$Data), by=1)
missing_dates_3<-date_range_3[!date_range_3 %in% isol_campinas$Data]

# calculando a media movel para x dias
dias = 14
mean_saopaulo <- rollmean(isol_saopaulo$Isol, dias, align = "right")
# graficos para inspeção
tempo <- (1:length(isol_saopaulo$Isol))
tempo_media <- (1:length(mean_saopaulo))

plot(isol_saopaulo$Isol, col="blue")
lines(mean_saopaulo, col="red", lwd = 2)
lines(predict(lm(isol_saopaulo$Isol~tempo)),col='blue',lty=2,lwd=2)
lines(predict(lm(mean_saopaulo~poly(tempo_media,9))),col='black',lwd=3)
#lines(predict(lm(isol_saopaulo$Isol~poly(tempo,2))),col='green',lwd=2)
#lines(predict(lm(isol_saopaulo$Isol~poly(tempo,3))),col='black',lwd=2)
lines(predict(lm(isol_saopaulo$Isol~poly(tempo,7))),col='green',lwd=4)
#
linearmodel1 <- lm(isol_saopaulo$Isol~tempo)
linearmodel2 <- lm(isol_saopaulo$Isol~poly(tempo,5))
#
#plot(isol_santos)
#plot(isol_campinas)
# -----------------
# salvando os dados
#write.csv(isol_saopaulo,'data/SaoPaulo_isolamento.csv')
#write.csv(isol_santos,'data/Santos_isolamento.csv')
#write.csv(isol_campinas,'data/Campinas_isolamento.csv')
#