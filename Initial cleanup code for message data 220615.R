## STEP 1: EXTRACTION and FITTING TO SERVE OUR NEEDS

# edit as you see fit
setwd('Desktop/CSAT')
df = read.table(list.files()[2], sep = '~', header = TRUE, allowEscapes = FALSE)
## When saving from Redshift, make sure to follow these settings:
# Go to Data: Save Data as
# Then make sure encoding is UTF-8, date/timestampnull/format is empty. Field delimiter is ~ and quote chatacter is '. Quote escape is escape, line ending is CRLF.
df$created_dt = as.Date(df$created_dt)
df$message = as.character(df$message)

distinctDF = df[!duplicated(df[c(2:9)]), ]
nondistinctdf = df[duplicated(df[c(2:9)]), ]


oneWeekData = distinctDF[distinctDF$created_dt >= '2016-06-13' & distinctDF$created_dt <= '2016-06-19', ]
plot(as.data.frame(table(distinctDF$created_dt))) # graph to show the number of comments a day
rm(oneWeekData, df, nondistinctdf)

sites = read.table(list.files()[7], sep = ",", header = TRUE, allowEscapes = FALSE)
sites$dw_eff_dt = as.Date(sites$dw_eff_dt)
sites = sites[order(sites$dw_eff_dt, decreasing = TRUE), ]
sitesUnique = sites[!duplicated(sites$post_id), ]

df = merge(sitesUnique, distinctDF, by = c("post_id"), all.y = TRUE)
write.csv(df, file = "comments_raw_merged_with_page_data_190716.csv")
