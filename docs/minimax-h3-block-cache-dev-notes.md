# MiniMax H3 Block Cache nativa per Raylight USP

Stato al 13 agosto 2026: implementazione funzionante su hardware reale, branch `feat-minimax-h3-block-cache`. Questa nota serve come promemoria per completare i test con calma prima di proporre la modifica al branch `dev`.

## Obiettivo e confini

La modifica porta in Raylight l'idea algoritmica di TE-Speed per MiniMax H3, senza copiarne il sistema di patching:

- nei passi `FULL` esegue tutti i blocchi H3 e salva il residuale locale `h_full - h_warm`;
- nei passi `CACHE` esegue soltanto il prefisso dei blocchi e applica il residuale del precedente passo `FULL`;
- il calcolo avviene dopo lo split USP e prima del gather;
- ogni rank conserva esclusivamente il proprio residuale locale, senza nuove comunicazioni distribuite;
- nessuna modifica a ComfyUI core, `patch_model.py`, `block_loop`, ConvRot, kernel quantizzati o pesi del modello;
- prima versione limitata a MiniMax H3 con Raylight USP; FSDP, EasyCache e TeaCache insieme alla Block Cache restano fuori scope.

Con Block Cache disabilitata, il forward segue il percorso Raylight preesistente.

## File interessati

Implementazione:

- `src/raylight/diffusion_models/minimax/block_cache.py`: configurazione, decisione FULL/CACHE, cache key, residuale locale, contatori, logging e cleanup;
- `src/raylight/diffusion_models/minimax/xdit_context_parallel.py`: integrazione nel block loop MiniMax H3 USP;
- `src/raylight/comfy_extra_dist/nodes_minimax_h3.py`: nodo `MiniMax H3 Block Cache (Ray USP)`;
- `src/raylight/distributed_worker/sampling_config.py`: invio esplicito della configurazione per ogni sampling, incluso OFF;
- `src/raylight/distributed_worker/ray_worker.py`: ricezione della configurazione fresca e rimozione dello stato precedente;
- `src/raylight/nodes.py` e `src/raylight/comfy_extra_dist/nodes_custom_sampler.py`: configurazione dei worker prima dei percorsi di sampling standard e custom.

Test:

- `tests/test_minimax_h3_block_cache.py`
- `tests/test_minimax_h3_block_loop.py`
- `tests/test_minimax_h3_block_cache_logging.py`
- `tests/test_minimax_h3_block_cache_disabled_logging.py`
- `tests/test_minimax_h3_block_cache_lifecycle.py`
- `tests/test_sampling_feature_config.py`

## Configurazione iniziale del nodo

- `sigma_threshold = 0.12`
- `start_percent = 0.10`
- `end_percent = 0.90`
- `max_cached_steps = 2`
- `cache_depth = 0.75`
- `debug = false` normalmente, `true` durante la verifica

Con 50 blocchi e profondità `0.75`, i passi CACHE eseguono un prefisso di 13 blocchi e ne saltano 37.

## Collegamento corretto del workflow

Il sampler custom recupera i Ray actors dal `RayCFGGuider`, non dal risultato del scheduler. Per questo l'uscita della Block Cache deve raggiungere entrambi:

```text
RayUNETLoader
    -> MiniMax H3 Block Cache
        -> RayBasicScheduler
        -> RayCFGGuider
```

Se si usa anche Sigma Shift:

```text
RayUNETLoader
    -> MiniMax H3 Sigma Shift
    -> MiniMax H3 Block Cache
        -> RayBasicScheduler
        -> RayCFGGuider
```

Collegare Block Cache soltanto al scheduler lascia il sampler sugli actors originali e produce un run baseline, anche se il nodo stampa `node enabled=True`.

## Lifecycle e protezione dallo stato obsoleto

Ogni sampling invia ai Ray workers una configurazione completa e nuova. L'assenza o la disattivazione del nodo viene tradotta esplicitamente in `enabled=False`; il forward abilita il percorso cache soltanto quando `config.get("enabled", False) is True`.

Il runtime del residuale nasce all'inizio del sampling ed è eliminato in `finally`, insieme a stream, contatori e riferimenti. Non rimane un residuale persistente sul modello o nei worker. La validità distingue conditioning/UUID e comprende shape dello shard, dtype, device e layout packed rilevante.

## Risultato del primo test hardware riuscito

Setup osservato: USP con due rank, 20 denoise step, parametri predefiniti del nodo e debug attivo.

```text
[H3 Block Cache][host] dispatch enabled=True workers=2
[H3 Block Cache][rank0] enabled=True threshold=0.12 start=0.10 end=0.90 mcs=2 depth=0.75
[H3 Block Cache][rank0] step=0 sigma=0.999992 mode=FULL prefix_blocks=50/50 cached_steps_count=0 residual_valid=False
[H3 Block Cache][rank0] step=2 sigma=0.990826 mode=CACHE prefix_blocks=13/50 cached_steps_count=1 residual_valid=True
[H3 Block Cache] FULL=8 CACHE=12 effective_blocks=556 skipped_blocks=444
```

Risultati:

- sequenza coerente `FULL -> CACHE -> CACHE -> FULL` nella finestra configurata;
- decisioni e sigma coerenti su rank 0 e rank 1;
- 8 passi FULL e 12 passi CACHE;
- `8 * 50 + 12 * 13 = 556` blocchi teorici eseguiti;
- `12 * 37 = 444` blocchi teorici evitati, pari al 44,4% dei 1000 blocchi-step baseline;
- tempo mostrato dalla progress bar per 20 step: circa `4:08 -> 2:24`, riduzione osservata di circa il 42% sul tratto misurato;
- il passo finale torna FULL fuori dalla finestra;
- le righe marcate da Ray come `[repeated ... across cluster]` sono deduplicazione dei log, non passi mancanti.

Il confronto temporale è indicativo: va ripetuto con stesso workflow, seed, stato di warm-up e condizioni di memoria.

## Verifiche automatiche già eseguite

Ultimo risultato registrato prima del test hardware:

```text
79 passed, 1 warning
```

Sono risultati puliti anche `py_compile` sui file modificati e `git diff --check`.

I test coprono il no-op OFF, decisioni FULL/CACHE, finestra start/end, sigma threshold, limite dei passi CACHE, separazione per UUID, invalidazione della firma, numero di blocchi, mancato prefetch dei blocchi saltati, logging, cleanup su eccezione, configurazione di tutti i sampler e regressione ON/OFF/ON con workers persistenti simulati.

## Checklist prima della proposta a `dev`

- Ripetere `ON -> OFF -> ON` senza riavviare Ray e conservare i tre log completi.
- In OFF verificare assenza di righe per-step CACHE e prestazioni equivalenti al baseline.
- Con stesso seed confrontare visivamente e, se pratico, numericamente output baseline e cache.
- Provare almeno I2V, REF2V e audio conditioning.
- Provare normal attention e SageAttention.
- Provare il modello/kernel quantizzato usato normalmente, controllando che nei passi CACHE non avvenga prefetch dei blocchi `k:50`.
- Verificare più schedule e numeri di step, inclusi limiti della finestra.
- Registrare VRAM di picco e tempi dopo warm-up; separare inizializzazione modello, denoise e VAE.
- Rieseguire l'intera suite immediatamente prima del commit.
- Controllare il diff finale ed escludere file o stampe diagnostiche non necessarie.

## Rischi e punti ancora da validare

- La qualità dipende dai parametri e dalla schedule; i valori predefiniti sono un punto di partenza, non una garanzia universale.
- Il test reale iniziale conferma MiniMax H3 USP a due rank, ma non copre ancora tutte le combinazioni I2V/REF2V/audio/attention/quantizzazione.
- FSDP non è implementato in questa versione.
- EasyCache e TeaCache simultanee devono rimanere incompatibili e fallire con un messaggio chiaro.
- Ray deduplica i log dei rank: per una diagnosi completa si può avviare Ray con `RAY_DEDUP_LOGS=0`, senza renderlo un requisito normale.

## Criterio per considerarla pronta

La modifica è pronta per una proposta a `dev` quando il test reale ON/OFF/ON dimostra che non esiste stato obsoleto negli stessi actors, il percorso OFF coincide con il baseline atteso, i workflow MiniMax H3 principali non mostrano regressioni e la suite completa resta verde.
