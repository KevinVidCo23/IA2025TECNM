import java.util.Scanner;

class Nodo {
    String nombre;
    Nodo izquierda, derecha;

    public Nodo(String nombre) {
        this.nombre = nombre;
        this.izquierda = null;
        this.derecha = null;
    }
}

class Arbol {
    Nodo raiz;

    public Arbol() {
        raiz = null;
    }

    // Verificar si el árbol está vacío
    public boolean vacio() {
        return raiz == null;
    }

    // Insertar un nodo en el árbol
    public void insertar(String nombre) {
        raiz = insertarRec(raiz, nombre);
    }

    private Nodo insertarRec(Nodo actual, String nombre) {
        if (actual == null) {
            return new Nodo(nombre);
        }
        if (nombre.compareTo(actual.nombre) < 0) {
            actual.izquierda = insertarRec(actual.izquierda, nombre);
        } else if (nombre.compareTo(actual.nombre) > 0) {
            actual.derecha = insertarRec(actual.derecha, nombre);
        }
        return actual;
    }

    // Buscar un nodo en el árbol
    public Nodo buscarNodo(String nombre) {
        return buscarRec(raiz, nombre);
    }

    private Nodo buscarRec(Nodo actual, String nombre) {
        if (actual == null || actual.nombre.equals(nombre)) {
            return actual;
        }
        if (nombre.compareTo(actual.nombre) < 0) {
            return buscarRec(actual.izquierda, nombre);
        } else {
            return buscarRec(actual.derecha, nombre);
        }
    }
}

public class ArbolBinario {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        Arbol arbol = new Arbol();
        int opcion;

        do {
            System.out.println("\n--- MENÚ ---");
            System.out.println("1. Insertar nodo");
            System.out.println("2. Buscar nodo");
            System.out.println("3. Verificar si el árbol está vacío");
            System.out.println("4. Salir");
            System.out.print("Selecciona una opción: ");
            opcion = sc.nextInt();
            sc.nextLine(); 

            switch (opcion) {
                case 1:
                    System.out.print("Ingresa el nombre a insertar: ");
                    String nombreInsertar = sc.nextLine();
                    arbol.insertar(nombreInsertar);
                    System.out.println("Nodo insertado con éxito.");
                    break;

                case 2:
                    System.out.print("Ingresa el nombre a buscar: ");
                    String nombreBuscar = sc.nextLine();
                    Nodo resultado = arbol.buscarNodo(nombreBuscar);
                    if (resultado != null) {
                        System.out.println("Nodo encontrado: " + resultado.nombre);
                    } else {
                        System.out.println("Nodo no encontrado.");
                    }
                    break;

                case 3:
                    if (arbol.vacio()) {
                        System.out.println("El árbol está vacío.");
                    } else {
                        System.out.println("El árbol NO está vacío.");
                    }
                    break;

                case 4:
                    System.out.println("Saliendo...");
                    break;

                default:
                    System.out.println("Opción inválida.");
                    break;
            }
        } while (opcion != 4);

        sc.close();
    }
}
