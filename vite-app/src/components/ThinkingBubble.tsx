import { BubbleContainer } from "./BubbleContainer";
import { Spinner } from "./Spinner";

export const ThinkingBubble = () => {
  return (
    <BubbleContainer role="thinking" showCopyButton={false}>
      <div className="flex items-center space-x-2">
        <Spinner size="sm" />
      </div>
    </BubbleContainer>
  );
};
