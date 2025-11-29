import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Stack;
import java.util.PriorityQueue;
import java.util.HashMap;

public class Arbol {
    Nodo raiz;
    

    public Arbol (Nodo raiz){
        this.raiz=raiz;
    }

    public Nodo realizarBusquedaEnAnchura(String objetivo){
        Queue<Nodo> cola = new LinkedList<Nodo>();
        HashSet<String> visitados = new HashSet<String>();
        cola.add(raiz);
        visitados.add(raiz.estado);
        boolean encontrado = false;
        Nodo actual = null;
        while(!encontrado && !cola.isEmpty()){
            actual = cola.poll();
            //función objetivo
            if(actual.estado.equals(objetivo)){
                encontrado = true;
            }else{
                List<String> sucesores = actual.obtenerSucesores();
                for (String sucesor : sucesores) {
                    if(visitados.contains(sucesor)) //si el nodo ya fue visitado, se ignora y no se agrega a la cola
                        continue;
                    System.err.println("Agregando a la cola => "+ sucesor);
                    cola.add(new Nodo(sucesor, actual));
                    visitados.add(sucesor);
                } 
            } 
        }
        return actual;
    }

    public Nodo realizarBusquedaEnProfundidad(String objetivo){
        Stack<Nodo> pila = new Stack<Nodo>();
        HashSet<String> visitados = new HashSet<String>();
        pila.push(raiz);
        visitados.add(raiz.estado);

        while(!pila.isEmpty()){
            Nodo actual = pila.pop();
            if(actual.estado.equals(objetivo)){
                return actual;
            }
            List<String> sucesores = actual.obtenerSucesores();
            for(String sucesor : sucesores){
                if(visitados.contains(sucesor))
                    continue;
                pila.push(new Nodo(sucesor, actual));
                visitados.add(sucesor);
            }
        }
        return null;
    }

    public Nodo realizarBusquedaCostoUniforme(String objetivo){
        PriorityQueue<Nodo> pq = new PriorityQueue<Nodo>((a,b) -> Integer.compare(a.costo, b.costo));
        HashMap<String, Integer> best = new HashMap<String, Integer>();

        pq.add(raiz);
        best.put(raiz.estado, raiz.costo);

        while(!pq.isEmpty()){
            Nodo actual = pq.poll();
            if(actual.estado.equals(objetivo)){
                return actual;
            }
            if(best.containsKey(actual.estado) && actual.costo > best.get(actual.estado))
                continue;

            List<String> sucesores = actual.obtenerSucesores();
            for(String sucesor : sucesores){
                int newCost = actual.costo + 1;
                if(!best.containsKey(sucesor) || newCost < best.get(sucesor)){
                    Nodo hijo = new Nodo(sucesor, actual);
                    pq.add(hijo);
                    best.put(sucesor, newCost);
                }
            }
        }
        return null;
    }

    
     //Heurística que favorece colocar y mantener en su lugar las fichas '7' y '8'.
    private int heuristica(String estado, String objetivo){
        int h = 0;
        for(int i = 0; i < estado.length(); i++){
            char c = estado.charAt(i);
            if(c == ' ') continue; 
            int posObj = objetivo.indexOf(c);
            if(posObj == -1) continue;
            int xi = i % 3;
            int yi = i / 3;
            int xj = posObj % 3;
            int yj = posObj / 3;
            int dist = Math.abs(xi - xj) + Math.abs(yi - yj);
            if(c == '7' || c == '8'){
                // peso mayor para 7 y 8 
                h += dist * 10;
            } else {
                h += dist;
            }
        }
        return h;
    }


    public Nodo realizarAEstrella(String objetivo){
        PriorityQueue<Nodo> pq = new PriorityQueue<Nodo>((a,b) -> Integer.compare(
            a.costo + heuristica(a.estado, objetivo),
            b.costo + heuristica(b.estado, objetivo)
        ));

        HashMap<String, Integer> best = new HashMap<String, Integer>();
        pq.add(raiz);
        best.put(raiz.estado, raiz.costo);

        while(!pq.isEmpty()){
            Nodo actual = pq.poll();
            if(actual.estado.equals(objetivo)){
                return actual;
            }

            // Si esta entrada tiene un coste peor que el mejor conocido, ignorar
            if(best.containsKey(actual.estado) && actual.costo > best.get(actual.estado))
                continue;

            List<String> sucesores = actual.obtenerSucesores();
            for(String sucesor : sucesores){
                int newCost = actual.costo + 1;
                if(!best.containsKey(sucesor) || newCost < best.get(sucesor)){
                    Nodo hijo = new Nodo(sucesor, actual);
                    pq.add(hijo);
                    best.put(sucesor, newCost);
                }
            }
        }
        return null;
    }
}

