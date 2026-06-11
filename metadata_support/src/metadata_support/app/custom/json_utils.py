import json
import logging
import re

logger = logging.getLogger(__name__)


class JsonUtils:
    """LLMレスポンスのJSON文字列を堅牢にパース／修復するユーティリティ。"""

    @classmethod
    def repair_and_parse(cls, resp_text, label=""):
        """LLMが返すJSON配列を修復してパースする。"""
        # コードブロック除去
        if resp_text.startswith("```"):
            resp_text = resp_text.split("\n", 1)[1] if "\n" in resp_text else resp_text[3:]
            if resp_text.endswith("```"):
                resp_text = resp_text[:-3]
            resp_text = resp_text.strip()
        resp_text = cls._remove_control_in_strings(resp_text)
        # 不正なエスケープシーケンスを修正
        resp_text = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', resp_text)
        resp_text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', resp_text)
        # 切り詰められたJSONを修復
        if not resp_text.rstrip().endswith(']'):
            last_complete = resp_text.rfind('},')
            last_end = resp_text.rfind('}')
            cut = max(last_complete, last_end)
            if cut > 0:
                resp_text = resp_text[:cut + 1] + '\n]'
                logger.warning("JSON_TRUNCATED_FIX | %s | cut_at=%d", label, cut)
        # `]` の後に余分なテキストがある場合は除去
        bracket_pos = resp_text.rfind(']')
        if bracket_pos >= 0 and resp_text[bracket_pos + 1:].strip():
            resp_text = resp_text[:bracket_pos + 1]
            logger.warning("JSON_EXTRA_DATA_FIX | %s", label)
        # エラーの逐次修復 (最大20回)
        for _fix in range(20):
            try:
                return json.loads(resp_text), resp_text
            except json.JSONDecodeError as je:
                err_msg = str(je).lower()
                if 'control character' in err_msg and je.pos > 0:
                    resp_text = resp_text[:je.pos] + ' ' + resp_text[je.pos + 1:]
                elif 'escape' in err_msg and je.pos > 0:
                    resp_text = resp_text[:je.pos] + '\\' + resp_text[je.pos:]
                elif ('unterminated' in err_msg) and je.pos > 0:
                    # 閉じクォートがない → 該当オブジェクトを切り捨て
                    last_complete = resp_text.rfind('},', 0, je.pos)
                    if last_complete > 0:
                        resp_text = resp_text[:last_complete + 1] + '\n]'
                        logger.warning("JSON_UNTERMINATED_FIX | %s | cut_at=%d", label, last_complete)
                    else:
                        raise
                elif 'delimiter' in err_msg or 'expecting' in err_msg:
                    resp_text = cls.fix_unescaped_quotes(resp_text)
                    try:
                        return json.loads(resp_text), resp_text
                    except json.JSONDecodeError:
                        pass
                    objs = cls.extract_objects(resp_text)
                    if objs:
                        logger.warning("JSON_FALLBACK_EXTRACT | %s | extracted=%d objects", label, len(objs))
                        return objs, resp_text
                    raise
                elif 'extra data' in err_msg and je.pos > 0:
                    resp_text = resp_text[:je.pos].strip()
                else:
                    raise
        return json.loads(resp_text), resp_text

    @staticmethod
    def _remove_control_in_strings(text):
        """JSON文字列値内の制御文字（\\x00-\\x1f）をスペースに置換する。"""
        result = []
        i = 0
        in_string = False
        while i < len(text):
            ch = text[i]
            if ch == '"' and (i == 0 or text[i - 1] != '\\'):
                in_string = not in_string
                result.append(ch)
            elif in_string and ord(ch) < 0x20:
                result.append(' ')
            else:
                result.append(ch)
            i += 1
        return ''.join(result)

    @staticmethod
    def fix_unescaped_quotes(text):
        """JSON文字列値内の未エスケープダブルクォートをエスケープする。"""
        result = []
        i = 0
        in_string = False
        prev_char = ''
        while i < len(text):
            ch = text[i]
            if ch == '"' and prev_char != '\\':
                if not in_string:
                    in_string = True
                    result.append(ch)
                else:
                    # 次の非空白文字を確認して、文字列終端かどうか判定
                    rest = text[i + 1:].lstrip()
                    if rest and rest[0] in (':', ',', '}', ']'):
                        in_string = False
                        result.append(ch)
                    elif not rest:
                        in_string = False
                        result.append(ch)
                    else:
                        # 文字列途中のクォート → エスケープ
                        result.append('\\"')
                        prev_char = '"'
                        i += 1
                        continue
            else:
                result.append(ch)
            prev_char = ch
            i += 1
        return ''.join(result)

    @staticmethod
    def extract_objects(text):
        """正規表現で個別JSONオブジェクトを抽出するフォールバック。"""
        objects = []
        for m in re.finditer(r'\{[^{}]*\}', text):
            try:
                obj = json.loads(m.group())
                objects.append(obj)
            except json.JSONDecodeError:
                continue
        return objects
