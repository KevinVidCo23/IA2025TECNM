import java.util.*;
public class DijkstraEjemplo {
    static int nodoConDistanciaMinima(int[] distancias, boolean[] visitado, int totalNodos) {
        int min = Integer.MAX_VALUE;
        int indiceMin = -1;
        for (int i = 0; i < totalNodos; i++) {
            if (!visitado[i] && distancias[i] <= min) {
                min = distancias[i];
                indiceMin = i;
            }
        }
        return indiceMin;
    }

    // Implementación del algoritmo de Dijkstra
    static void dijkstra(int[][] grafo, int nodoInicial) {
        int totalNodos = grafo.length;
        int[] distancias = new int[totalNodos];        
        boolean[] visitado = new boolean[totalNodos]; 

        Arrays.fill(distancias, Integer.MAX_VALUE);
        distancias[nodoInicial] = 0; 

        for (int contador = 0; contador < totalNodos - 1; contador++) {
            int nodoActual = nodoConDistanciaMinima(distancias, visitado, totalNodos);
            visitado[nodoActual] = true;

            for (int vecino = 0; vecino < totalNodos; vecino++) {
                if (!visitado[vecino] && grafo[nodoActual][vecino] != 0 &&
                    distancias[nodoActual] != Integer.MAX_VALUE &&
                    distancias[nodoActual] + grafo[nodoActual][vecino] < distancias[vecino]) {
                    distancias[vecino] = distancias[nodoActual] + grafo[nodoActual][vecino];
                }
            }
        }

        // Imprimir resultados
        System.out.println("Nodo\tDistancia desde el nodo " + nodoInicial);
        for (int i = 0; i < totalNodos; i++) {
            System.out.println(i + "\t" + distancias[i]);
        }
    }

    public static void main(String[] args) {
        int[][] grafo = {
            {0, 10, 0, 30, 100},
            {10, 0, 50, 0, 0},
            {0, 50, 0, 20, 10},
            {30, 0, 20, 0, 60},
            {100, 0, 10, 60, 0}
        };

        dijkstra(grafo, 0); 
    }
}
