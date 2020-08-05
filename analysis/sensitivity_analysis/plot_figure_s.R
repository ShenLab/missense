args=(commandArgs(TRUE))
for(i in 1:length(args)){
	eval(parse(text=args[[i]]))
}


chd = read.table(chd_path, header=T, sep=",")
autism = read.table(asd_path, header=T, sep=",")
controls = read.table(control_path, header=T, sep=",")


plotComparison <- function(x, y, zz, output_dir, method="", genes="All") {
	x = x[!is.na(x)]
	y = y[!is.na(y)]
	zz = zz[!is.na(zz)]

	
	output = paste(output_dir, "/case-vs-control_", "MVP_", genes, ".separate.pdf", sep="")
	pdf(output, width=4, height=4)
	par(family="sans", mar=c(4,4,1,0.5))

	hist(x, freq=F, xlab="Rank score", ylim=range(c(0,1.5)),  border="white", main="")
	#plot(c(), c(), xlab="Rank score", ylab="Density", ylim=range(c(0, 1.5)), xlim=range(c(0, 1.0)), main="", border="white")
	lines(density(x, from=0, to=1), col='red', lwd=2)
	grid(col="gray")
	lines(density(y, from=0, to=1), col='purple', lwd=2)
	lines(density(zz, from=0, to=1), col='blue', lwd=2)
	title(paste(method), line = 0)

	legend("bottomleft", c("CHD", "ASD", "Controls"), col=c("red", "purple", "blue"), lwd=2)
	wtest1 = wilcox.test(x, zz)
	wtest2 = wilcox.test(y, zz)
	print(wtest1)
	print(wtest2)
	legend("topleft", paste("CHD vs controls: p=", signif(wtest1$p.value, 2), "\n",  "ASD vs controls: p=", signif(wtest2$p.value, 2), sep=""), bty="n")
	dev.off()

}


plotComparison( chd$MVP_rank, autism$MVP_rank,  controls$MVP_rank, output_dir, paste("MVP (block_num=",block_num, ")"), "All" )
