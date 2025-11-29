import java.util.Scanner;
import java.util.Stack;
public class puzzle8 {

    

    public static void main(String a[]) {
        Scanner scanner = new Scanner(System.in);
        // valores por defecto
        String estadoInicialDefault = "483657 12";
        String estadoFinalDefault = " 12345678";

        System.out.println("Estado inicial (presione Enter para usar valor por defecto: \""+estadoInicialDefault+"\"):");
        String entrada = scanner.nextLine();
        String estadoInicial = entrada.isEmpty() ? estadoInicialDefault : entrada;

        System.out.println("Estado objetivo (presione Enter para usar valor por defecto: \""+estadoFinalDefault+"\"):");
        entrada = scanner.nextLine();
        String estadoFinal = entrada.isEmpty() ? estadoFinalDefault : entrada;

        Arbol arbol = new Arbol(new Nodo(estadoInicial, null));

        System.out.println("Seleccione algoritmo: 1) Primero en anchura  2) Primero en profundidad  3) Costo uniforme  4) Eurística");
        int opcion = 1;
        try{
            opcion = Integer.parseInt(scanner.nextLine());
        }catch(Exception e){
            opcion = 1;
        }

        Nodo resultado = null;
        switch(opcion){
            case 1:
                resultado = arbol.realizarBusquedaEnAnchura(estadoFinal);
                break;
            case 2:
                resultado = arbol.realizarBusquedaEnProfundidad(estadoFinal);
                break;
            case 3:
                resultado = arbol.realizarBusquedaCostoUniforme(estadoFinal);
                break;
            case 4:
                resultado = arbol.realizarAEstrella(estadoFinal);
                break;
            default:
                System.out.println("Opción inválida. Saliendo.");
                System.exit(0);
        }

        if(resultado == null){
            System.out.println("No se encontró solución.");
            System.exit(0);
        }

        Stack<Nodo> pilaCamino = new Stack<>();
        Nodo actual = resultado;

        while(actual!=null){
            pilaCamino.push(actual);
            actual = actual.padre;
        }

        int pCont = 1;
        System.out.println("============== Camino de solución: ==============");
        while(!pilaCamino.isEmpty()){
            Nodo paso = pilaCamino.pop();
            System.out.println("Paso #"+pCont);
            imprimirEstado(paso.estado);
            System.out.println();
            pCont++;
        }

        
        
    }

    private static void imprimirEstado(String estado){
        for (int i = 0; i < estado.length(); i++) {
        System.out.print(estado.charAt(i) + " ");
        if ((i + 1) % 3 == 0) {
            System.out.println();
            }
        }
    }
}