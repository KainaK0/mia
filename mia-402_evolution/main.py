import random
import copy

# --- 1. CONFIGURACIÓN DE RECURSOS Y DATOS ---

# Diccionario de capacidades maximas por recurso
CAPACIDAD_RECURSOS = {
    'mechanical': 200,
    'electrical': 110,
    'welder': 70,
    'grua_220': 4,
    'grua_250': 3,  # Recurso crítico
    'camion_grua': 8
}

# Definición de actividades (Data ampliada)
# ID 0 es Inicio Ficticio, ID 13 es Fin Ficticio
datos_mantenimiento = {
    0: {'nombre': 'Inicio', 'dur': 0, 'pred': [], 'rec': {}},
    1: {'nombre': 'Desconexion Elec.', 'dur': 4, 'pred': [0], 
        'rec': {'mechanical': 2, 'electrical': 10, 'camion_grua': 1}},
    2: {'nombre': 'Desmontaje Tuberia', 'dur': 6, 'pred': [0], 
        'rec': {'mechanical': 15, 'welder': 5, 'grua_220': 1, 'camion_grua': 1}},
    3: {'nombre': 'Izaje Cubierta', 'dur': 8, 'pred': [], 
        'rec': {'mechanical': 10, 'grua_250': 1}}, # Requiere la grúa crítica
    4: {'nombre': 'Inspeccion Rotor', 'dur': 5, 'pred': [3], 
        'rec': {'mechanical': 5, 'electrical': 5}},
    5: {'nombre': 'Reparacion Sold A', 'dur': 10, 'pred': [3], 
        'rec': {'mechanical': 5, 'welder': 20, 'camion_grua': 1}},
    6: {'nombre': 'Reparacion Sold B', 'dur': 10, 'pred': [3], 
        'rec': {'mechanical': 5, 'welder': 20}},
    7: {'nombre': 'Cambio Rodamientos', 'dur': 6, 'pred': [4], 
        'rec': {'mechanical': 12, 'camion_grua': 1}},
    8: {'nombre': 'Alineamiento Eje', 'dur': 8, 'pred': [7], 
        'rec': {'mechanical': 8, 'electrical': 2}},
    9: {'nombre': 'Montaje Rotor', 'dur': 8, 'pred': [5, 6, 8], 
        'rec': {'mechanical': 15, 'welder': 5, 'grua_250': 1}}, # Requiere la grúa crítica otra vez
    10: {'nombre': 'Conexion Final', 'dur': 5, 'pred': [9], 
        'rec': {'mechanical': 2, 'electrical': 15, 'camion_grua': 1}},
    11: {'nombre': 'Montaje Tuberia', 'dur': 6, 'pred': [9], 
        'rec': {'mechanical': 15, 'welder': 10, 'grua_220': 1, 'camion_grua': 1}},
    12: {'nombre': 'Pruebas Carga', 'dur': 4, 'pred': [10, 11], 
        'rec': {'mechanical': 5, 'electrical': 5}},
    13: {'nombre': 'Fin', 'dur': 0, 'pred': [12], 'rec': {}}
}


# --- 1. CONFIGURACIÓN MASIVA DE RECURSOS ---
CAPACIDAD_RECURSOS = {
    'mechanical': 150,
    'welder': 60,
    'electrician': 60,
    'instrumentation': 20, # Recurso muy escaso
    'safety_insp': 10,     # Cuello de botella administrativo
    'scaffolder': 30,      # Necesario para iniciar trabajos en altura
    'crane_300t': 1,       # Recurso CRITICO único
    'crane_50t': 4,
    'forklift': 10
}

# --- 2. GENERADOR DE ESCENARIOS (DATA FACTORY) ---
def generar_datos_masivos(num_actividades=50):
    """
    Genera un proyecto realista con fases lógicas para evitar
    grafos aleatorios sin sentido.
    """
    print(f"--- Generando escenario complejo con {num_actividades} actividades ---")
    
    datos = {}
    # Nodo Inicio
    datos[0] = {'nombre': 'Inicio Proyecto', 'dur': 0, 'rec': {}, 'pred': []}
    
    # Fases del proyecto para dar estructura
    # 0-30%: Desmontaje (Mucha grúa, mecánicos)
    # 30-70%: Reparación (Soldadura, instrumentación, espera piezas)
    # 70-100%: Montaje y Pruebas (Grúa, eléctrico, safety)
    
    for i in range(1, num_actividades + 1):
        fase = i / num_actividades
        
        # Definir Duración (aleatoria con sesgo)
        duracion = random.randint(2, 15)
        if random.random() < 0.1: duracion += 10 # Tareas largas ocasionales
        
        # Definir Recursos según fase
        recursos = {}
        
        # Lógica de asignación probabilística basada en la fase
        if fase < 0.3: # FASE DESMONTAJE
            nom = f"Desmontaje Bloque {i}"
            recursos['mechanical'] = random.randint(5, 20)
            recursos['scaffolder'] = random.randint(0, 5)
            if random.random() < 0.2: recursos['crane_50t'] = 1
            if random.random() < 0.05: recursos['crane_300t'] = 1 # Uso raro de grúa gigante
            
        elif fase < 0.7: # FASE REPARACIÓN
            nom = f"Reparación/Soldadura {i}"
            recursos['mechanical'] = random.randint(2, 10)
            recursos['welder'] = random.randint(5, 15)
            if random.random() < 0.3: recursos['safety_insp'] = 1 # Permisos de fuego
            if random.random() < 0.2: recursos['instrumentation'] = 2
            
        else: # FASE MONTAJE Y PRUEBAS
            nom = f"Montaje y Test {i}"
            recursos['mechanical'] = random.randint(5, 15)
            recursos['electrician'] = random.randint(5, 10)
            recursos['safety_insp'] = 1 # Casi siempre requiere validación
            if random.random() < 0.1: recursos['crane_300t'] = 1 # Montaje final pesado
            
        # Limpieza de recursos (quitar ceros)
        recursos = {k:v for k,v in recursos.items() if v > 0}
        
        # Definir Predecesores (DAG - Grafo Acíclico Dirigido)
        # Solo se puede depender de tareas con ID menor para evitar ciclos
        preds = []
        if i == 1:
            preds = [0]
        else:
            # Conectar con 1 a 3 tareas anteriores aleatorias
            # Favorecer tareas recientes para crear cadenas
            num_preds = random.randint(1, 3)
            candidatos = list(range(max(0, i-10), i)) # Ventana de precedencia local
            preds = random.sample(candidatos, min(len(candidatos), num_preds))
            
        datos[i] = {'nombre': nom, 'dur': duracion, 'rec': recursos, 'pred': preds}

    # Nodo Fin
    datos[num_actividades + 1] = {
        'nombre': 'Fin Proyecto', 'dur': 0, 'rec': {}, 
        'pred': [i for i in range(1, num_actividades + 1)] # Depende de todo (simplificado)
    }
    
    return datos


# --- 2. CLASES Y FUNCIONES DEL ALGORITMO ---

class Individuo:
    def __init__(self, num_tareas):
        self.genes = [random.random() for _ in range(num_tareas)]
        self.makespan = float('inf')
        self.cronograma = {}

def verificar_disponibilidad(tiempo_inicio, duracion, demanda_tarea, uso_recursos_global):
    """
    Verifica si hay suficientes recursos de TODOS los tipos necesarios
    desde tiempo_inicio hasta tiempo_inicio + duracion.
    """
    for t in range(tiempo_inicio, tiempo_inicio + duracion):
        for tipo_rec, cantidad_necesaria in demanda_tarea.items():
            if cantidad_necesaria == 0: continue
            
            # Cuánto se está usando ya en este instante 't'
            uso_actual = uso_recursos_global.get(t, {}).get(tipo_rec, 0)
            capacidad_max = CAPACIDAD_RECURSOS.get(tipo_rec, 0)
            
            if uso_actual + cantidad_necesaria > capacidad_max:
                return False # No cabe
    return True

def actualizar_uso_recursos(tiempo_inicio, duracion, demanda_tarea, uso_recursos_global):
    """
    Reserva los recursos en la matriz global de tiempo.
    """
    for t in range(tiempo_inicio, tiempo_inicio + duracion):
        if t not in uso_recursos_global:
            uso_recursos_global[t] = {}
        for tipo_rec, cantidad in demanda_tarea.items():
            uso_actual = uso_recursos_global[t].get(tipo_rec, 0)
            uso_recursos_global[t][tipo_rec] = uso_actual + cantidad

def serial_sgs_multi(genes, tareas):
    n = len(tareas)
    lista_prioridad = sorted(range(n), key=lambda k: genes[k], reverse=True)
    
    cronograma = {} 
    fin_tareas = {id_t: 0 for id_t in tareas} # Inicializar tiempos de fin
    recursos_ocupados = {} # {tiempo: {'mech': 10, 'elec': 5...}}
    
    completed = set()
    
    # Loop principal de programación
    while len(completed) < n:
        candidata = -1
        
        # Buscar tarea elegible con mayor prioridad
        for tarea_id in lista_prioridad:
            if tarea_id not in completed:
                preds = tareas[tarea_id]['pred']
                # Verificar si predecesores terminaron (ignorando -1 o nulos)
                if all(p in completed for p in preds):
                    candidata = tarea_id
                    break
        
        if candidata == -1: break 

        # Determinar inicio más temprano por precedencias (CPM simple)
        preds = tareas[candidata]['pred']
        inicio_minimo = 0
        if preds:
            inicio_minimo = max([fin_tareas[p] for p in preds])
        
        duracion = tareas[candidata]['dur']
        demanda = tareas[candidata]['rec']
        
        # Buscar primer hueco con RECURSOS disponibles
        t = inicio_minimo
        encontrado = False
        while not encontrado:
            if verificar_disponibilidad(t, duracion, demanda, recursos_ocupados):
                # Agendar
                cronograma[candidata] = t
                fin_tareas[candidata] = t + duracion
                actualizar_uso_recursos(t, duracion, demanda, recursos_ocupados)
                completed.add(candidata)
                encontrado = True
            else:
                t += 1 # Desplazar un momento en el tiempo y probar de nuevo
                
    makespan = max(fin_tareas.values())
    return makespan, cronograma

def busqueda_local(individuo):
    # Swap simple para intentar escapar de optimos locales
    if random.random() > 0.5: return individuo
    
    vecino = copy.deepcopy(individuo)
    idx1, idx2 = random.sample(range(len(vecino.genes)), 2)
    vecino.genes[idx1], vecino.genes[idx2] = vecino.genes[idx2], vecino.genes[idx1]
    
    mk, _ = serial_sgs_multi(vecino.genes, datos_mantenimiento)
    
    if mk < individuo.makespan:
        vecino.makespan = mk
        return vecino
    return individuo

def algoritmo_memetico_rcpsp(generaciones=1000, poblacion_size=30, datos_mantenimiento = {}):
    num_tareas = len(datos_mantenimiento)
    poblacion = [Individuo(num_tareas) for _ in range(poblacion_size)]
    
    # Eval inicial
    for ind in poblacion:
        mk, cro = serial_sgs_multi(ind.genes, datos_mantenimiento)
        ind.makespan = mk
        ind.cronograma = cro
        
    mejor_global = min(poblacion, key=lambda x: x.makespan)
    print(f"Inicio: Mejor Makespan encontrado = {mejor_global.makespan} horas")

    for g in range(generaciones):
        nueva_pob = [copy.deepcopy(mejor_global)] # Elitismo
        
        while len(nueva_pob) < poblacion_size:
            # Torneo
            p1 = min(random.sample(poblacion, 4), key=lambda x: x.makespan)
            p2 = min(random.sample(poblacion, 4), key=lambda x: x.makespan)
            
            # Cruce
            hijo = Individuo(num_tareas)
            cut = random.randint(1, num_tareas-1)
            hijo.genes = p1.genes[:cut] + p2.genes[cut:]
            
            # Mutacion
            if random.random() < 0.2:
                idx = random.randint(0, num_tareas-1)
                hijo.genes[idx] = random.random()
            
            # Evaluar
            mk, cro = serial_sgs_multi(hijo.genes, datos_mantenimiento)
            hijo.makespan = mk
            hijo.cronograma = cro
            
            # Memético (Mejora local)
            hijo = busqueda_local(hijo)
            
            nueva_pob.append(hijo)
            
        poblacion = nueva_pob
        mejor_gen = min(poblacion, key=lambda x: x.makespan)
        if mejor_gen.makespan < mejor_global.makespan:
            mejor_global = copy.deepcopy(mejor_gen)
            print(f"Gen {g}: Nueva mejora encontrada -> {mejor_global.makespan} horas")
            
    return mejor_global

# --- EJECUCIÓN ---
if __name__ == "__main__":
    datos_mantenimiento = generar_datos_masivos(200)
    solucion = algoritmo_memetico_rcpsp(datos_mantenimiento = datos_mantenimiento)
    
    print("\n" + "="*60)
    print(f"MEJOR CRONOGRAMA ENCONTRADO (Makespan: {solucion.makespan} hrs)")
    print("="*60)
    print(f"{'ID':<3} | {'ACTIVIDAD':<20} | {'INICIO':<6} | {'FIN':<6} | {'RECURSOS CLAVE'}")
    print("-" * 60)
    
    cron_ordenado = sorted(solucion.cronograma.items(), key=lambda x: x[1])
    
    for tid, inicio in cron_ordenado:
        if tid == 0 or tid == 13: continue # Omitir dummies visualmente
        dat = datos_mantenimiento[tid]
        fin = inicio + dat['dur']
        
        # Formatear string de recursos para visualización
        recs_str = ", ".join([f"{k}:{v}" for k,v in dat['rec'].items() if v > 0])
        if len(recs_str) > 30: recs_str = recs_str[:27] + "..."
        
        print(f"{tid:<3} | {dat['nombre']:<20} | {inicio:<6} | {fin:<6} | {recs_str}")