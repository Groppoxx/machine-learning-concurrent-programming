package lr

import (
    "runtime"
    "sync"
)

// partial guarda las sumas parciales de un bloque de datos.
type partial struct {
    sumX, sumY, sumXY, sumX2 float64
}

// chunkBounds divide el tamaño total en numChunks rangos [lo, hi).
func chunkBounds(length, numChunks int) [][2]int {
    size := length / numChunks
    bounds := make([][2]int, numChunks)
    for i := 0; i < numChunks; i++ {
        lo := i * size
        hi := lo + size
        if i == numChunks-1 {
            hi = length // último trozo puede ser un poco más grande
        }
        bounds[i] = [2]int{lo, hi}
    }
    return bounds
}

// computePartial recorre x[lo:hi] e y[lo:hi] y acumula sus sumas.
func computePartial(x, y []float64, lo, hi int) partial {
    var p partial
    for j := lo; j < hi; j++ {
        p.sumX += x[j]
        p.sumY += y[j]
        p.sumXY += x[j] * y[j]
        p.sumX2 += x[j] * x[j]
    }
    return p
}

// gatherPartials lanza una goroutine por cada rango y recoge los partials.
func gatherPartials(x, y []float64, bounds [][2]int) []partial {
    numChunks := len(bounds)
    parts := make([]partial, numChunks)
    var wg sync.WaitGroup

    for i, b := range bounds {
        wg.Add(1)
        go func(idx, lo, hi int) {
            defer wg.Done()
            parts[idx] = computePartial(x, y, lo, hi)
        }(i, b[0], b[1])
    }

    wg.Wait() // esperar a que todas terminen
    return parts
}

// reducePartials suma todos los partials en totales globales.
func reducePartials(parts []partial) (sumX, sumY, sumXY, sumX2 float64) {
    for _, p := range parts {
        sumX += p.sumX
        sumY += p.sumY
        sumXY += p.sumXY
        sumX2 += p.sumX2
    }
    return
}

// computeCoefficientsConcurrent aplica la fórmula de regresión.
func computeCoefficientsConcurrent(nTotal, sumX, sumY, sumXY, sumX2 float64) (m, b float64) {
    m = (nTotal*sumXY - sumX*sumY) / (nTotal*sumX2 - sumX*sumX)
    b = (sumY - m*sumX) / nTotal
    return
}

// ConcurrentLinearRegression orquesta todo el cálculo en paralelo.
func ConcurrentLinearRegression(x, y []float64) (m, b float64) {
    nTotal := float64(len(x))
    numCPU := runtime.GOMAXPROCS(0)        // número de hilos

    bounds := chunkBounds(len(x), numCPU) // rangos de trabajo
    parts := gatherPartials(x, y, bounds) // sumar en paralelo
    sumX, sumY, sumXY, sumX2 := reducePartials(parts) // reducción

    return computeCoefficientsConcurrent(nTotal, sumX, sumY, sumXY, sumX2)
}
