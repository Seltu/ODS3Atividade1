using System.Collections;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using TMPro;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

public class SageDialogue : MonoBehaviour
{
    [Header("UI")]
    [SerializeField] private SpriteRenderer npcSprite;
    [SerializeField] private TMP_InputField inputField;
    [SerializeField] private Button inputButton;
    [SerializeField] private TMP_Text output;

    [Header("Sprites por Expressão")]
    [SerializeField] private ExpressionSprite[] expressions; // mapeie "Neutral", "Confident", etc.

    // ==== Modelo de request/response (compatível com sua API atual) ====
    [System.Serializable] class AskReq { public string question; public int k = 5; }
    [System.Serializable] class AskResp { public string reply; } // ignoramos citations

    // ==== Estrutura interna para cada pedaço ====
    [System.Serializable] public class Segment { public string text; public string expression; }

    private readonly List<Segment> _queue = new();
    private int _idx = 0;
    private bool _isPlaying = false;

    // Regex: captura "frase ... [Expressao]" no final da linha
    private static readonly Regex TagAtEnd = new Regex(@"^(.*?)\s*\[([A-Za-z]+)\]\s*$", RegexOptions.Compiled);

    void Awake()
    {
        // Um único botão: envia quando não está tocando; avança quando está.
        inputButton.onClick.RemoveAllListeners();
        inputButton.onClick.AddListener(OnButtonClick);
        SetButtonLabel("Enviar");
    }

    // Botão principal
    private void OnButtonClick()
    {
        if (_isPlaying)
        {
            NextSegment();
        }
        else
        {
            StartCoroutine(AskServer(inputField.text));
        }
    }

    // === Fluxo de rede ===
    IEnumerator AskServer(string q)
    {
        if (string.IsNullOrWhiteSpace(q)) yield break;

        // Desativa botão durante requisição
        inputButton.interactable = false;
        inputField.interactable = false;

        var req = JsonUtility.ToJson(new AskReq { question = q });
        using var www = new UnityWebRequest("http://localhost:8000/ask", "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(req);
        www.uploadHandler = new UploadHandlerRaw(bodyRaw);
        www.downloadHandler = new DownloadHandlerBuffer();
        www.SetRequestHeader("Content-Type", "application/json");
        SetNpcFace("Sad");

        yield return www.SendWebRequest();

        inputButton.interactable = true;
        inputField.interactable = true;

        if (www.result != UnityWebRequest.Result.Success)
        {
            output.text = "Sorry, i couldn't hear that, could you say it again? (ERROR)";
            SetNpcFace("Neutral");
            yield break;
        }

        var resp = JsonUtility.FromJson<AskResp>(www.downloadHandler.text);
        PrepareQueueFromReply(resp.reply);
        BeginPlayback();
    }

    // Converte o reply em uma fila de segmentos (texto + expressão)
    private void PrepareQueueFromReply(string reply)
    {
        Debug.Log(reply);
        _queue.Clear();
        _idx = 0;

        if (string.IsNullOrWhiteSpace(reply)) return;

        // Monta alternância com as expressões válidas do Inspector (Neutral|Confident|...)
        var alt = new System.Text.StringBuilder();
        for (int i = 0; i < expressions.Length; i++)
        {
            if (i > 0) alt.Append("|");
            alt.Append(Regex.Escape(expressions[i].expression));
        }

        // Qualquer [Expressao] (case-insensitive)
        var tagPattern = new Regex(@"\[(" + alt.ToString() + @")\]", RegexOptions.IgnoreCase);

        // Parágrafos: se houver quebras de linha, usa; senão trata tudo como um só
        var chunks = reply.Contains("\n") ? reply.Split('\n') : new string[] { reply };

        foreach (var raw in chunks)
        {
            var line = raw.Trim();
            if (string.IsNullOrEmpty(line)) continue;

            // Pega a primeira tag válida do parágrafo (se houver)
            var m = tagPattern.Match(line);
            string expr = "Neutral";
            if (m.Success)
            {
                string found = m.Groups[1].Value.Trim();
                // normaliza para o nome exatamente configurado no Inspector
                foreach (var map in expressions)
                {
                    if (map.expression.Equals(found, System.StringComparison.OrdinalIgnoreCase))
                    {
                        expr = map.expression;
                        break;
                    }
                }
            }

            // Remove TODAS as tags [Expressao] do texto mostrado
            string cleaned = tagPattern.Replace(line, "").Trim();
            // Compacta espaços duplos que podem sobrar
            cleaned = Regex.Replace(cleaned, @"\s{2,}", " ");

            if (!string.IsNullOrEmpty(cleaned))
                _queue.Add(new Segment { text = cleaned, expression = expr });
        }

        // Fallback: se nada foi gerado, adiciona tudo como Neutral (sem tags)
        if (_queue.Count == 0)
        {
            string cleanedAll = tagPattern.Replace(reply, "").Trim();
            if (!string.IsNullOrEmpty(cleanedAll))
                _queue.Add(new Segment { text = cleanedAll, expression = "Neutral" });
        }
    }



    // Inicia a reprodução dos segmentos
    private void BeginPlayback()
    {
        _isPlaying = true;
        inputField.gameObject.SetActive(false);
        SetButtonLabel("Próximo");

        if (_queue.Count > 0)
        {
            ShowSegment(_queue[0]);
        }
        else
        {
            FinishPlayback();
        }
    }

    // Avança para o próximo
    private void NextSegment()
    {
        _idx++;
        if (_idx >= _queue.Count)
        {
            FinishPlayback();
            return;
        }
        ShowSegment(_queue[_idx]);
    }

    // Exibe um segmento (troca sprite + texto)
    private void ShowSegment(Segment s)
    {
        SetNpcFace(s.expression);
        output.text = s.text; // aqui você pode colocar letras-per-second, etc.
    }

    // Termina a reprodução
    private void FinishPlayback()
    {
        _isPlaying = false;
        inputField.gameObject.SetActive(true);
        inputField.text = string.Empty;
        SetButtonLabel("Enviar");
        // Mantém o último texto no output
    }

    // === Auxiliares ===
    private void SetButtonLabel(string label)
    {
        var txt = inputButton.GetComponentInChildren<TMP_Text>();
        if (txt != null) txt.text = label;
    }

    private void SetNpcFace(string expression)
    {
        // procura mapeamento; se não achar, usa o primeiro sprite ou mantém
        foreach (var map in expressions)
        {
            if (map.expression.Equals(expression, System.StringComparison.OrdinalIgnoreCase))
            {
                if (map.sprite != null) npcSprite.sprite = map.sprite;
                return;
            }
        }
        // fallback opcional: usar uma expressão "Neutral" se existir
        foreach (var map in expressions)
        {
            if (map.expression.Equals("Neutral", System.StringComparison.OrdinalIgnoreCase) && map.sprite != null)
            {
                npcSprite.sprite = map.sprite;
                return;
            }
        }
    }

    [System.Serializable]
    public struct ExpressionSprite
    {
        public string expression; // ex.: "Neutral", "Confident", "Annoyed"
        public Sprite sprite;
    }
}
