package lr

// computeSums recorre los datos y acumula sumas básicas.
func computeSums(x, y []float64) (sumX, sumY, sumXY, sumX2 float64) {
    for i := range x {
        sumX += x[i]        // suma de x
        sumY += y[i]        // suma de y
        sumXY += x[i] * y[i]// suma de x*y
        sumX2 += x[i] * x[i]// suma de x^2
    }
    return
}

// computeCoefficients calcula la pendiente m y el intercepto b.
func computeCoefficients(n, sumX, sumY, sumXY, sumX2 float64) (m, b float64) {
    m = (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)
    b = (sumY - m*sumX) / n
    return
}

// LinearRegression orquesta el cálculo de m y b usando las dos funciones anteriores.
func LinearRegression(x, y []float64) (m, b float64) {
    n := float64(len(x))                       // número de puntos
    sumX, sumY, sumXY, sumX2 := computeSums(x, y)
    return computeCoefficients(n, sumX, sumY, sumXY, sumX2)
}
